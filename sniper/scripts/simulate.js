#!/usr/bin/env node
/**
 * Dry-run a four-figure snipe across the launch conditions that plausibly show up.
 *
 * The chain is simulated; everything else is the real thing -- the real Sniper, the real
 * guards in the real order, the real bigint math, the real CoW order construction. So
 * this tests the decision path at size. It is NOT evidence about live Base: pool
 * reserves here are assumed, not observed. Use scripts/fork-test.js for that.
 *
 *   BUY_USD=1000 node scripts/simulate.js
 */
import { ethers } from 'ethers';
import { createMockChain } from '../test/mock-chain.js';
import { createMockCow } from '../test/mock-cow.js';
import { loadConfig } from '../src/config.js';
import { buildContracts } from '../src/wiring.js';
import { preflight } from '../src/preflight.js';
import { Sniper } from '../src/sniper.js';
import { ExitManager, buildRungs } from '../src/exit.js';
import { CowClient } from '../src/cow.js';
import { LAPTOP, WETH } from '../src/constants.js';

const BUY_USD = process.env.BUY_USD ?? '1000';

// Size against the live price when a Base RPC is reachable. A stale assumption here is
// not cosmetic: it changes how much ETH the buy is, and therefore the price impact.
async function liveEthUsd() {
  const rpc = process.env.BASE_RPC;
  if (!rpc) return null;
  try {
    const { proxiedProvider } = await import('./_proxy.js');
    const { CHAINLINK_ETH_USD, CHAINLINK_ABI } = await import('../src/constants.js');
    const p = proxiedProvider(rpc);
    const feed = new ethers.Contract(CHAINLINK_ETH_USD, CHAINLINK_ABI, p);
    const [, answer] = await feed.latestRoundData();
    const decimals = Number(await feed.decimals());
    p.destroy();
    return BigInt(Math.round(Number(ethers.formatUnits(answer, decimals))));
  } catch {
    return null;
  }
}

const live = await liveEthUsd();
const ETH_USD = live ?? BigInt(process.env.ETH_USD ?? '2479');
const SUPPLY = 1_000_000_000n;               // $LAPTOP total supply
const POOL = '0x1111111111111111111111111111111111111111';
const AERO_POOL = '0x3333333333333333333333333333333333333333';
const quiet = { log() {}, error() {} };

const fmt = (wei, d = 18) => ethers.formatUnits(wei, d);
const usd = (wei) => (Number(ethers.formatEther(wei)) * Number(ETH_USD)).toFixed(2);

/** Token reserve implied by a pool's WETH depth and a fully-diluted valuation. */
function tokenReserveFor({ wethEth, fdvUsd }) {
  const priceUsdPerToken = Number(fdvUsd) / Number(SUPPLY);
  const poolUsd = Number(wethEth) * Number(ETH_USD);
  return ethers.parseUnits((poolUsd / priceUsdPerToken).toFixed(6), 18);
}

async function runScenario(sc) {
  const reserves = {
    weth: ethers.parseEther(String(sc.wethEth)),
    token: tokenReserveFor(sc)
  };
  const key = (sc.venue === 'aero' ? AERO_POOL : POOL).toLowerCase();

  const chainState = {
    walletBalance: ethers.parseEther('5'),
    ethUsdPrice: ETH_USD * 100_000_000n,   // 8-decimal feed, matching ETH_USD above
    poolWeth: { [key]: reserves.weth },
    poolReserves: { [key]: reserves },
    ...(sc.venue === 'aero'
      ? { aeroPool: AERO_POOL }
      : { uniPools: { 3000: POOL } })
  };

  // Override only the sell direction for the honeypot cases.
  if (sc.sellBack !== undefined) {
    const cp = (amountIn) => {
      const afterFee = amountIn * 9970n / 10_000n;
      return (afterFee * reserves.token) / (reserves.weth + afterFee);
    };
    chainState.uniQuote = ({ tokenIn, amountIn }) =>
      tokenIn.toLowerCase() === WETH.toLowerCase() ? cp(amountIn) : sc.sellBack;
  }

  const mock = createMockChain(chainState);
  const provider = await mock.listen();
  const config = loadConfig({
    BASE_WSS: 'wss://sim', PRIVATE_KEY: ethers.Wallet.createRandom().privateKey,
    BUY_USD, EARLIEST_BUY: '2020-01-01T00:00:00Z', LIVE: 'true',
    MIN_TOKENS_OUT: sc.minTokensOut ?? '1',
    MIN_WETH_LIQ: String(sc.minWethLiq ?? 30),
    MAX_PRICE_IMPACT_BPS: String(sc.maxImpactBps ?? 300)
  });
  const wallet = new ethers.Wallet(config.PRIVATE_KEY, provider);
  const contracts = buildContracts(provider, wallet);

  await preflight({ provider, wallet, contracts, config });
  mock.state.tokenBalances[wallet.address.toLowerCase()] = 0n;

  const sniper = new Sniper({ contracts, wallet, config, logger: quiet });
  sniper.setTokenDecimals(18);

  let result;
  try {
    result = await sniper.scan(1);
  } catch (err) {
    result = { status: 'error', reason: err.shortMessage ?? err.message };
  }

  provider.destroy();
  await mock.close();
  return { result, config, reserves };
}

console.log(`Simulated snipe of $${BUY_USD} at ETH $${ETH_USD}` +
  (live ? ' (live Chainlink)' : ' (assumed - set BASE_RPC for the live price)'));
console.log('Chain is simulated; guards, math and order construction are the real code.\n');

const scenarios = [
  { label: 'thin launch pool',      wethEth: 3,   fdvUsd: 80_000_000,  minWethLiq: 1 },
  { label: 'modest pool',           wethEth: 10,  fdvUsd: 80_000_000,  minWethLiq: 1 },
  { label: 'at MIN_WETH_LIQ=30',    wethEth: 30,  fdvUsd: 80_000_000 },
  { label: 'deep pool',             wethEth: 100, fdvUsd: 80_000_000 },
  { label: 'very deep',             wethEth: 400, fdvUsd: 80_000_000 },
  { label: 'below threshold',       wethEth: 12,  fdvUsd: 80_000_000, minWethLiq: 30 },
  { label: 'Aerodrome venue',       wethEth: 60,  fdvUsd: 80_000_000, venue: 'aero' },
  { label: 'honeypot: no sell',     wethEth: 60,  fdvUsd: 80_000_000, sellBack: null },
  { label: 'punitive sell tax',     wethEth: 60,  fdvUsd: 80_000_000,
    sellBack: ethers.parseEther('0.02') },
  { label: 'launched 5x too high',  wethEth: 60,  fdvUsd: 400_000_000,
    minTokensOut: '2800' }
];

const rows = [];
for (const sc of scenarios) {
  const { result, config } = await runScenario(sc);
  const impact = result.impact ?? result.op?.priceImpactBps;
  rows.push({
    label: sc.label,
    depth: `${sc.wethEth} WETH`,
    outcome: result.status === 'bought' ? 'BUY'
      : result.status === 'blocked' ? `blocked: ${result.reason}`
      : result.status,
    impact: impact !== undefined && impact !== null ? `${(Number(impact) / 100).toFixed(2)}%` : '-',
    tokens: result.op?.quotedOut ? Number(fmt(result.op.quotedOut)).toLocaleString('en-US',
      { maximumFractionDigits: 0 }) : '-',
    spend: `$${usd(config.BUY_WEI)}`
  });
}

const pad = (s, n) => String(s).padEnd(n);
console.log(pad('scenario', 22) + pad('depth', 11) + pad('outcome', 26) + pad('impact', 9) + pad('tokens', 12) + 'spend');
console.log('-'.repeat(90));
for (const r of rows) {
  console.log(pad(r.label, 22) + pad(r.depth, 11) + pad(r.outcome, 26) + pad(r.impact, 9) + pad(r.tokens, 12) + r.spend);
}

// --- what MIN_TOKENS_OUT means in valuation terms --------------------------------
console.log(`\nMIN_TOKENS_OUT for $${BUY_USD}, by the launch valuation you are willing to pay:`);
console.log(pad('  FDV', 16) + pad('price/token', 16) + 'tokens for your buy');
console.log('  ' + '-'.repeat(52));
for (const fdv of [50, 100, 250, 500, 1000, 2000]) {
  const price = (fdv * 1e6) / Number(SUPPLY);
  const tokens = Number(BUY_USD) / price;
  console.log('  ' + pad(`$${fdv}M`, 14) + pad(`$${price.toFixed(4)}`, 16) +
    tokens.toLocaleString('en-US', { maximumFractionDigits: 0 }));
}
console.log('\n  Set MIN_TOKENS_OUT to the row you refuse to pay above. Above ~$500M FDV the');
console.log('  bot should decline: that is the guard doing its job, not a failure.');

// --- full lifecycle: buy, then rest the exit ladder ------------------------------
console.log('\n' + '='.repeat(90));
console.log('Full lifecycle: $' + BUY_USD + ' buy into a 60 WETH pool, then the exit ladder');
console.log('='.repeat(90));

const lifecycle = await (async () => {
  const reserves = {
    weth: ethers.parseEther('60'),
    token: tokenReserveFor({ wethEth: 60, fdvUsd: 80_000_000 })
  };
  const mock = createMockChain({
    walletBalance: ethers.parseEther('5'),
    ethUsdPrice: ETH_USD * 100_000_000n,
    uniPools: { 3000: POOL },
    poolWeth: { [POOL.toLowerCase()]: reserves.weth },
    poolReserves: { [POOL.toLowerCase()]: reserves }
  });
  const provider = await mock.listen();
  const config = loadConfig({
    BASE_WSS: 'wss://sim', PRIVATE_KEY: ethers.Wallet.createRandom().privateKey,
    BUY_USD, EARLIEST_BUY: '2020-01-01T00:00:00Z', LIVE: 'true',
    MIN_TOKENS_OUT: '8000', MIN_WETH_LIQ: '30', MAX_PRICE_IMPACT_BPS: '300',
    EXIT_LADDER: process.env.EXIT_LADDER ?? '2x:50,3x:30,5x:20'
  });
  const wallet = new ethers.Wallet(config.PRIVATE_KEY, provider);
  const contracts = buildContracts(provider, wallet);
  await preflight({ provider, wallet, contracts, config });

  const sniper = new Sniper({ contracts, wallet, config, logger: quiet });
  sniper.setTokenDecimals(18);
  const buy = await sniper.scan(1);

  console.log(`\nentry:`);
  console.log(`  spend            $${usd(config.BUY_WEI)}  (${fmt(config.BUY_WEI)} ETH)`);
  console.log(`  venue            ${buy.op.venue} fee ${buy.op.fee}`);
  console.log(`  price impact     ${(Number(buy.op.priceImpactBps) / 100).toFixed(2)}%`);
  console.log(`  tokens quoted    ${Number(fmt(buy.op.quotedOut)).toLocaleString('en-US', { maximumFractionDigits: 0 })}`);
  console.log(`  amountOutMinimum ${Number(fmt(sniper.minOutFor(buy.op))).toLocaleString('en-US', { maximumFractionDigits: 0 })}`);
  console.log(`  round-trip kept  ${(Number(buy.op.retentionBps) / 100).toFixed(2)}%`);

  // Suppose the fill matched the quote.
  const held = buy.op.quotedOut;
  const rungs = buildRungs({ totalTokens: held, entryWei: config.BUY_WEI, ladder: config.EXIT_LADDER });

  const cow = createMockCow();
  const baseUrl = await cow.listen();
  mock.state.tokenBalances[wallet.address.toLowerCase()] = held;
  const manager = new ExitManager({
    client: new CowClient({ chainId: 8453, baseUrl }),
    wallet, token: contracts.laptopWrite,
    sellToken: LAPTOP, buyToken: WETH, logger: quiet
  });
  const placed = await manager.placeLadder({
    totalTokens: held, entryWei: config.BUY_WEI, ladder: config.EXIT_LADDER
  });

  console.log(`\nexit ladder (${config.EXIT_LADDER_SPEC}), resting on CoW:`);
  let proceeds = 0n;
  for (const r of rungs) {
    proceeds += r.buyAmount;
    console.log(`  ${r.label.padEnd(8)} sell ${Number(fmt(r.sellAmount)).toLocaleString('en-US', { maximumFractionDigits: 0 }).padStart(9)}` +
      ` tokens for ${fmt(r.buyAmount).slice(0, 8).padStart(9)} ETH  ($${usd(r.buyAmount)})`);
  }
  console.log(`\n  orders placed    ${placed.placed.length}`);
  console.log(`  sig rejections   ${cow.rejections.length}`);
  console.log(`  if all fill      $${usd(proceeds)} on $${usd(config.BUY_WEI)} in ` +
    `(${(Number(proceeds * 100n / config.BUY_WEI) / 100).toFixed(2)}x)`);
  console.log(`  gross profit     $${(Number(usd(proceeds)) - Number(usd(config.BUY_WEI))).toFixed(2)} ` +
    `before gas and CoW fees`);

  provider.destroy(); await mock.close(); await cow.close();
})();

