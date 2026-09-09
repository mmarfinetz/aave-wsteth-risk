#!/usr/bin/env node
/**
 * How much liquidity does a given buy actually need?
 *
 * Two parts. First it measures real Base pools: for each, the quoter is asked for the
 * full buy and for a 1/100th probe, and the gap between the two rates is the true price
 * impact. That measured figure is compared against what constant-product predicts from
 * the pool's WETH balance, which says whether the model can be trusted to extrapolate.
 *
 * Then it inverts the model: for a target impact, what depth is required.
 *
 *   BASE_RPC=https://... BUY_USD=1000 node scripts/liquidity-curve.js
 */
import { ethers } from 'ethers';
import { proxiedProvider } from './_proxy.js';
import { buildContracts } from '../src/wiring.js';
import { readEthUsd, usdToWei, priceImpactBps } from '../src/pricing.js';
import { WETH, UNI_FEE_TIERS, AERO_FACTORY, ERC20_ABI } from '../src/constants.js';

const RPC = process.env.BASE_RPC ?? 'https://mainnet.base.org';
const BUY_USD = process.env.BUY_USD ?? '1000';
// A liquid token whose WETH pools span a wide range of depths, so one buy size can be
// measured against thin and deep books alike.
const REF = ethers.getAddress(process.env.REF_TOKEN ?? '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913');

const provider = proxiedProvider(RPC);
const wallet = ethers.Wallet.createRandom().connect(provider);
const contracts = buildContracts(provider, wallet);

/** Constant-product impact for spending `dx` into a pool holding `x` on the input side. */
const modelImpactBps = (dx, x) => (dx * 10_000n) / (x + dx);
/** Inverse: depth needed on the input side to keep impact at or under `bps`. */
const depthForImpact = (dx, bps) => (dx * (10_000n - bps)) / bps;

try {
  const feed = await readEthUsd(contracts.ethUsdFeed, { maxAgeSec: 86_400 });
  const buyWei = usdToWei(BUY_USD, feed);
  const ethUsd = Number(ethers.formatUnits(feed.price, feed.decimals));
  const probe = buyWei / 100n;
  const token = new ethers.Contract(REF, ERC20_ABI, provider);
  const decimals = Number(await token.decimals());
  const symbol = await token.symbol();

  console.log(`Measuring against real Base pools at block ${await provider.getBlockNumber()}`);
  console.log(`buy $${BUY_USD} = ${Number(ethers.formatEther(buyWei)).toFixed(4)} ETH ` +
    `(ETH $${ethUsd.toFixed(2)}), reference token ${symbol}\n`);

  const pad = (s, n) => String(s).padEnd(n);
  console.log(pad('venue', 14) + pad('pool WETH', 14) + pad('measured', 11) +
    pad('model', 11) + 'cost on the buy');
  console.log('-'.repeat(70));

  const rows = [];
  for (const fee of UNI_FEE_TIERS) {
    const pool = await contracts.uniFactory.getPool(WETH, REF, fee);
    if (pool === ethers.ZeroAddress) continue;
    const depth = await contracts.weth.balanceOf(pool);
    if (depth === 0n) continue;
    try {
      const [full] = await contracts.uniQuoter.quoteExactInputSingle.staticCall({
        tokenIn: WETH, tokenOut: REF, amountIn: buyWei, fee, sqrtPriceLimitX96: 0n
      });
      const [small] = await contracts.uniQuoter.quoteExactInputSingle.staticCall({
        tokenIn: WETH, tokenOut: REF, amountIn: probe, fee, sqrtPriceLimitX96: 0n
      });
      const measured = priceImpactBps({ probeIn: probe, probeOut: small, actualIn: buyWei, actualOut: full });
      const model = modelImpactBps(buyWei, depth);
      rows.push({ venue: `uni ${fee}`, depth, measured, model });
    } catch { /* uninitialized tier */ }
  }

  const aero = await contracts.aeroFactory.getPool(WETH, REF, false);
  if (aero !== ethers.ZeroAddress) {
    const depth = await contracts.weth.balanceOf(aero);
    const routes = [{ from: WETH, to: REF, stable: false, factory: AERO_FACTORY }];
    const full = (await contracts.aeroRouterRead.getAmountsOut(buyWei, routes)).at(-1);
    const small = (await contracts.aeroRouterRead.getAmountsOut(probe, routes)).at(-1);
    const measured = priceImpactBps({ probeIn: probe, probeOut: small, actualIn: buyWei, actualOut: full });
    rows.push({ venue: 'aerodrome', depth, measured, model: modelImpactBps(buyWei, depth) });
  }

  rows.sort((a, b) => (a.depth < b.depth ? -1 : 1));
  for (const r of rows) {
    const costUsd = (Number(r.measured) / 10_000) * Number(BUY_USD);
    console.log(
      pad(r.venue, 14) +
      pad(Number(ethers.formatEther(r.depth)).toFixed(2), 14) +
      pad(`${(Number(r.measured) / 100).toFixed(2)}%`, 11) +
      pad(`${(Number(r.model) / 100).toFixed(2)}%`, 11) +
      `$${costUsd.toFixed(2)}`
    );
  }

  console.log('\nUniswap V3 concentrates liquidity, so measured impact comes in at or below');
  console.log('what constant product predicts from the raw WETH balance. A brand-new launch');
  console.log('pool is usually full-range, where the two converge — so the model is the');
  console.log('conservative read, and that is the one worth sizing against.\n');

  console.log(`Depth required for a $${BUY_USD} buy, by impact you are willing to eat:`);
  console.log(pad('  target impact', 18) + pad('WETH needed', 16) + pad('pool value', 16) + 'cost of the buy');
  console.log('  ' + '-'.repeat(62));
  for (const bps of [50n, 100n, 200n, 300n, 500n, 1000n]) {
    const need = depthForImpact(buyWei, bps);
    const needEth = Number(ethers.formatEther(need));
    console.log(
      '  ' + pad(`${Number(bps) / 100}%`, 16) +
      pad(needEth.toFixed(1), 16) +
      pad(`$${(needEth * ethUsd).toLocaleString('en-US', { maximumFractionDigits: 0 })}`, 16) +
      `$${((Number(bps) / 10_000) * Number(BUY_USD)).toFixed(2)}`
    );
  }

  console.log('\nRound trip (in and out) roughly doubles the impact, and the pool fee is paid');
  console.log('on each leg. On a 1% tier that is 2% in fees alone before impact.');
} catch (err) {
  console.error(err.shortMessage ?? err.message ?? err);
  process.exitCode = 1;
} finally {
  provider.destroy();
}
