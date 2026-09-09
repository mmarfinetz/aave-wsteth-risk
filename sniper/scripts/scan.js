#!/usr/bin/env node
/**
 * One read-only scan of live Base: what pools exist, what the bot would decide, and how
 * far off the launch gate is. Sends nothing, signs nothing, needs no key.
 *
 *   BASE_RPC=https://... npm run scan
 */
import 'dotenv/config';
import { ethers } from 'ethers';
import { proxiedProvider } from './_proxy.js';
import { buildContracts } from '../src/wiring.js';
import { Sniper } from '../src/sniper.js';
import { readEthUsd, usdToWei } from '../src/pricing.js';
import { LAPTOP, WETH, UNI_FEE_TIERS, AERO_FACTORY, ERC20_ABI } from '../src/constants.js';

const RPC = process.env.BASE_RPC ?? 'https://mainnet.base.org';
const TOKEN = ethers.getAddress(process.env.TOKEN ?? LAPTOP);
const BUY_USD = process.env.BUY_USD ?? '1000';
const MIN_WETH_LIQ = process.env.MIN_WETH_LIQ ?? '30';
const EARLIEST = Date.parse(process.env.EARLIEST_BUY ?? '2026-09-09T00:00:00-04:00');

const provider = proxiedProvider(RPC);
const wallet = ethers.Wallet.createRandom().connect(provider);
const contracts = buildContracts(provider, wallet);
const token = new ethers.Contract(TOKEN, ERC20_ABI, provider);

const dur = (ms) => {
  const s = Math.abs(Math.round(ms / 1000));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return `${h}h ${m}m`;
};

try {
  const [block, decimals, symbol] = await Promise.all([
    provider.getBlockNumber(), token.decimals(), token.symbol()
  ]);
  const feed = await readEthUsd(contracts.ethUsdFeed, { maxAgeSec: 86_400 });
  const buyWei = usdToWei(BUY_USD, feed);

  console.log(`Base block ${block}   ${new Date().toISOString()}`);
  console.log(`token   ${symbol} ${TOKEN}`);
  console.log(`ETH     $${ethers.formatUnits(feed.price, feed.decimals)} ` +
    `(${feed.ageSec}s old)  ->  $${BUY_USD} = ${Number(ethers.formatEther(buyWei)).toFixed(4)} ETH`);
  const delta = EARLIEST - Date.now();
  console.log(`gate    EARLIEST_BUY ${new Date(EARLIEST).toISOString()} ` +
    `(${delta > 0 ? dur(delta) + ' away' : 'passed ' + dur(delta) + ' ago'})`);
  console.log(`floor   MIN_WETH_LIQ ${MIN_WETH_LIQ} WETH\n`);

  console.log('pools');
  let anyLiquidity = false;
  for (const fee of UNI_FEE_TIERS) {
    const pool = await contracts.uniFactory.getPool(WETH, TOKEN, fee);
    if (pool === ethers.ZeroAddress) { console.log(`  uni ${String(fee).padStart(5)}   -`); continue; }
    const liq = await contracts.weth.balanceOf(pool);
    let quote = '';
    try {
      const [out] = await contracts.uniQuoter.quoteExactInputSingle.staticCall({
        tokenIn: WETH, tokenOut: TOKEN, amountIn: buyWei, fee, sqrtPriceLimitX96: 0n
      });
      quote = `quotes ${Number(ethers.formatUnits(out, decimals)).toLocaleString('en-US',
        { maximumFractionDigits: 0 })} ${symbol}`;
      anyLiquidity = true;
    } catch (err) {
      quote = `quote reverts (${(err.shortMessage ?? err.message ?? '').slice(0, 40)})`;
    }
    console.log(`  uni ${String(fee).padStart(5)}   ${pool}  ${ethers.formatEther(liq).padStart(22)} WETH  ${quote}`);
  }
  const aero = await contracts.aeroFactory.getPool(WETH, TOKEN, false);
  if (aero === ethers.ZeroAddress) console.log('  aerodrome   -');
  else {
    const liq = await contracts.weth.balanceOf(aero);
    console.log(`  aerodrome   ${aero}  ${ethers.formatEther(liq).padStart(22)} WETH`);
    anyLiquidity = true;
  }

  // What the real decision path says, right now.
  const config = {
    BUY_WEI: buyWei, MIN_WETH_LIQ: ethers.parseEther(MIN_WETH_LIQ),
    SLIPPAGE_BPS: 500n, LIVE: false, EARLIEST_BUY_MS: EARLIEST,
    REQUIRE_SELL_PATH: true, MAX_PRICE_IMPACT_BPS: 300n, MAX_SCAN_STALENESS_MS: 60_000
  };
  const sniper = new Sniper({ contracts, wallet, config, token: TOKEN, logger: { log() {}, error() {} } });
  sniper.setTokenDecimals(decimals, symbol);
  const result = await sniper.scan(block);

  console.log(`\ndecision  ${result.status}${result.reason ? ` (${result.reason})` : ''}`);
  if (result.status === 'before-earliest-buy') {
    console.log('          the time gate is holding it; liquidity is not being considered yet');
  }
  if (result.op) {
    console.log(`          best venue ${result.op.venue}, quote ` +
      `${Number(ethers.formatUnits(result.op.quotedOut, decimals)).toLocaleString('en-US')} ${symbol}`);
  }
  if (!anyLiquidity) console.log('          no tradable pool exists yet');
} catch (err) {
  console.error(err.shortMessage ?? err.message ?? err);
  process.exitCode = 1;
} finally {
  provider.destroy();
}
