#!/usr/bin/env node
/**
 * Run the real buy path against real Base contracts, on a local fork.
 *
 * This is the only check in this repo that is not talking to a mock. Everything under
 * test/ validates that the code is self-consistent with assumptions this repo made up;
 * here the Uniswap factory, the QuoterV2, the Aerodrome router and the pool state are
 * the actual deployed ones, at real reserves. If the struct layouts or addresses are
 * wrong, this is where it shows.
 *
 * Setup (needs foundry and an archive-capable Base RPC):
 *
 *   anvil --fork-url https://mainnet.base.org --port 8545
 *   FORK_RPC=http://127.0.0.1:8545 TOKEN=0x<liquid token> node scripts/fork-test.js
 *
 * Point TOKEN at a token that ALREADY has liquidity. Before launch the real target has
 * no pool, so a fork of today proves only that discovery returns nothing. Proving the
 * buy executes requires a token that trades -- any established Base token will do, and
 * the fork spends nothing real.
 */
import { ethers } from 'ethers';
import { buildContracts } from '../src/wiring.js';
import { Sniper } from '../src/sniper.js';
import { inspectUniswapV3, inspectAerodrome } from '../src/venues.js';
import { WETH, ERC20_ABI, BASE_CHAIN_ID } from '../src/constants.js';

const FORK_RPC = process.env.FORK_RPC ?? 'http://127.0.0.1:8545';
const TOKEN = process.env.TOKEN;
const BUY_ETH = process.env.BUY_ETH ?? '0.05';
const MIN_WETH_LIQ = process.env.MIN_WETH_LIQ ?? '1';

if (!TOKEN) {
  console.error('Set TOKEN to a Base token address that already has liquidity.');
  process.exit(1);
}

const provider = new ethers.JsonRpcProvider(FORK_RPC, undefined, { staticNetwork: true });

const fail = (m) => { console.error(`FAIL  ${m}`); process.exitCode = 1; };
const ok = (m) => console.log(`ok    ${m}`);
const info = (m) => console.log(`..    ${m}`);

try {
  const network = await provider.getNetwork();
  if (network.chainId !== BASE_CHAIN_ID) {
    throw new Error(`Fork reports chain ${network.chainId}, expected Base ${BASE_CHAIN_ID}. ` +
      'Is anvil forking Base?');
  }
  ok(`fork is Base ${network.chainId} at block ${await provider.getBlockNumber()}`);

  // Refuse to run against anything that is not a local fork -- these are real routers.
  if (!/127\.0\.0\.1|localhost/.test(FORK_RPC)) {
    throw new Error(`FORK_RPC must be a local fork, got ${FORK_RPC}. This script sends transactions.`);
  }

  const wallet = ethers.Wallet.createRandom().connect(provider);
  await provider.send('anvil_setBalance', [wallet.address, ethers.toBeHex(ethers.parseEther('100'))]);
  ok(`funded a throwaway wallet with 100 fork ETH: ${wallet.address}`);

  const token = ethers.getAddress(TOKEN);
  const contracts = buildContracts(provider, wallet);
  const erc20 = new ethers.Contract(token, ERC20_ABI, provider);

  const [symbol, decimals] = await Promise.all([erc20.symbol(), erc20.decimals()]);
  ok(`token ${token} is ${symbol}, ${decimals} decimals (read from the real contract)`);

  const buyWei = ethers.parseEther(BUY_ETH);
  const diagnostics = [];
  const ctx = {
    ...contracts, token, buyWei,
    minWethLiq: ethers.parseEther(MIN_WETH_LIQ), diagnostics
  };

  // Real factories, real quoter, real reserves.
  const [uni, aero] = await Promise.all([inspectUniswapV3(ctx), inspectAerodrome(ctx)]);
  const found = [...uni, ...aero];

  if (found.length === 0) {
    for (const d of diagnostics) info(JSON.stringify(d, (_, v) => typeof v === 'bigint' ? v.toString() : v));
    throw new Error('No venue cleared MIN_WETH_LIQ on the fork. Pick a more liquid TOKEN, ' +
      'or lower MIN_WETH_LIQ.');
  }

  for (const op of found) {
    ok(`${op.venue}${op.fee ? ` fee ${op.fee}` : ''} pool ${op.pool}: ` +
       `${ethers.formatEther(op.wethLiquidity)} WETH, quotes ` +
       `${ethers.formatUnits(op.quotedOut, decimals)} ${symbol} for ${BUY_ETH} ETH`);
  }

  // Now the real thing: execute the buy through the actual router.
  const config = {
    BUY_WEI: buyWei,
    MIN_WETH_LIQ: ethers.parseEther(MIN_WETH_LIQ),
    SLIPPAGE_BPS: 500n,
    LIVE: true,
    EARLIEST_BUY_MS: 0,
    REQUIRE_SELL_PATH: true,
    MAX_PRICE_IMPACT_BPS: BigInt(process.env.MAX_PRICE_IMPACT_BPS ?? '2000'),
    MAX_SCAN_STALENESS_MS: 60_000
  };

  const sniper = new Sniper({ contracts, wallet, config, token });
  sniper.setTokenDecimals(decimals);
  sniper.absoluteMinOut = 1n;   // the fork test is about mechanics, not price

  const before = await erc20.balanceOf(wallet.address);
  const result = await sniper.scan(await provider.getBlockNumber());

  if (result.status !== 'bought') {
    fail(`scan returned "${result.status}"${result.reason ? ` (${result.reason})` : ''}`);
  } else {
    const after = await erc20.balanceOf(wallet.address);
    const gained = after - before;
    ok(`buy executed through the real router: tx ${result.txHash}`);
    ok(`wallet received ${ethers.formatUnits(gained, decimals)} ${symbol}`);
    if (gained <= 0n) fail('transaction succeeded but no tokens arrived — check for a transfer hook');
    const slippageFloor = (result.op.quotedOut * 9500n) / 10_000n;
    if (gained < slippageFloor) {
      fail(`received less than the slippage floor (${ethers.formatUnits(slippageFloor, decimals)})`);
    } else {
      ok('received at least amountOutMinimum');
    }
    if (result.op.priceImpactBps !== undefined) {
      info(`price impact estimate: ${Number(result.op.priceImpactBps) / 100}%`);
    }
  }

  console.log(process.exitCode ? '\nFork test FAILED.' : '\nFork test passed against real Base contracts.');
} catch (err) {
  fail(err.shortMessage ?? err.message ?? String(err));
} finally {
  provider.destroy();
}
