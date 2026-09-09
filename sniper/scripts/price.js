#!/usr/bin/env node
/**
 * What is this token worth, and is that number backed by anything?
 *
 * Reports three things that can disagree badly for a token this new:
 *   1. the price implied by each pool's own state (slot0), which is whatever the pool
 *      was seeded at and nothing more;
 *   2. what an aggregator quotes across sizes, which looks authoritative but is only a
 *      quote, not a fill;
 *   3. how much of the token actually sits in pools, which is what any of it could
 *      really be traded against.
 *
 * A flat aggregator price across sizes reads like depth but is the opposite tell: real
 * AMM liquidity always degrades with size. Flat means the quote is coming from somewhere
 * other than routable pools.
 *
 *   BASE_RPC=https://... npm run price
 */
import { ethers } from 'ethers';
import { proxiedProvider } from './_proxy.js';
import { buildContracts } from '../src/wiring.js';
import { readEthUsd } from '../src/pricing.js';
import { CowClient } from '../src/cow.js';
import { LAPTOP, WETH, UNI_FEE_TIERS, ERC20_ABI } from '../src/constants.js';

const RPC = process.env.BASE_RPC ?? 'https://mainnet.base.org';
const TOKEN = ethers.getAddress(process.env.TOKEN ?? LAPTOP);
const USDC = ethers.getAddress('0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913');

const POOL_ABI = [
  'function slot0() view returns (uint160 sqrtPriceX96,int24 tick,uint16 a,uint16 b,uint16 d,uint8 e,bool unlocked)',
  'function liquidity() view returns (uint128)',
  'function token0() view returns (address)'
];

const provider = proxiedProvider(RPC);
const contracts = buildContracts(provider, ethers.Wallet.createRandom().connect(provider));
const token = new ethers.Contract(TOKEN, ERC20_ABI, provider);

// The CoW client needs a proxy-aware fetch in sandboxed environments; direct otherwise.
const PROXY = process.env.HTTPS_PROXY ?? process.env.https_proxy;
let fetchImpl = fetch;
if (PROXY) {
  const { fetch: undiciFetch, ProxyAgent } = await import('undici');
  const dispatcher = new ProxyAgent(PROXY);
  fetchImpl = (url, opts = {}) => undiciFetch(url, { ...opts, dispatcher });
}

try {
  const [feed, decimals, symbol, supply, block] = await Promise.all([
    readEthUsd(contracts.ethUsdFeed, { maxAgeSec: 86_400 }),
    token.decimals(), token.symbol(), token.totalSupply(), provider.getBlockNumber()
  ]);
  const ethUsd = Number(ethers.formatUnits(feed.price, feed.decimals));
  const supplyN = Number(ethers.formatUnits(supply, decimals));
  const fdv = (usdPer) => `$${((usdPer * supplyN) / 1e6).toLocaleString('en-US', { maximumFractionDigits: 0 })}M`;

  console.log(`${symbol} ${TOKEN}`);
  console.log(`block ${block}   ETH $${ethUsd.toFixed(2)}   supply ${supplyN.toLocaleString()}\n`);

  console.log('1. price each pool has set for itself');
  let inventory = 0n;
  for (const fee of UNI_FEE_TIERS) {
    const pool = await contracts.uniFactory.getPool(WETH, TOKEN, fee);
    if (pool === ethers.ZeroAddress) continue;
    const p = new ethers.Contract(pool, POOL_ABI, provider);
    const [wethBal, tokBal] = await Promise.all([
      contracts.weth.balanceOf(pool), token.balanceOf(pool)
    ]);
    inventory += tokBal;
    let price = 'uninitialized';
    try {
      const [sqrtPriceX96] = await p.slot0();
      if (sqrtPriceX96 > 0n) {
        const token0 = await p.token0();
        const r = Number(sqrtPriceX96) / Number(2n ** 96n);
        const t1PerT0 = r * r;
        const tokensPerEth = token0.toLowerCase() === WETH.toLowerCase() ? t1PerT0 : 1 / t1PerT0;
        const usdPer = (1 / tokensPerEth) * ethUsd;
        price = `$${usdPer.toPrecision(4)}/token  FDV ${fdv(usdPer)}`;
      }
    } catch { price = 'slot0 unavailable'; }
    console.log(`   uni ${String(fee).padStart(5)}  ${ethers.formatEther(wethBal).slice(0, 12).padStart(12)} WETH  ` +
      `${Number(ethers.formatUnits(tokBal, decimals)).toLocaleString().padStart(12)} ${symbol}  ${price}`);
  }
  const usdcPool = await contracts.uniFactory.getPool(USDC, TOKEN, 10_000);
  if (usdcPool !== ethers.ZeroAddress) {
    const bal = await token.balanceOf(usdcPool);
    inventory += bal;
    console.log(`   uni/USDC  ${' '.repeat(17)}${Number(ethers.formatUnits(bal, decimals)).toLocaleString().padStart(12)} ${symbol}`);
  }

  console.log('\n2. what an aggregator quotes, by size');
  const client = new CowClient({ chainId: 8453, fetchImpl });
  const from = ethers.Wallet.createRandom().address;
  let biggestQuoteTokens = 0;
  for (const eth of ['0.004', '0.04', '0.4', '4']) {
    try {
      const r = await client.quoteSell({
        sellToken: WETH, buyToken: TOKEN,
        sellAmountBeforeFee: ethers.parseEther(eth), from
      });
      const out = Number(ethers.formatUnits(r.quote.buyAmount, decimals));
      biggestQuoteTokens = Math.max(biggestQuoteTokens, out);
      const usdPer = (Number(eth) * ethUsd) / out;
      console.log(`   ${(eth + ' ETH').padEnd(9)} $${(Number(eth) * ethUsd).toFixed(0).padStart(6)}  ` +
        `-> ${out.toFixed(2).padStart(12)} ${symbol}  $${usdPer.toFixed(4)}/token  FDV ${fdv(usdPer)}`);
    } catch (err) {
      const b = err.body ?? {};
      console.log(`   ${(eth + ' ETH').padEnd(9)} ${b.errorType ?? 'error'}: ${(b.description ?? err.message).slice(0, 50)}`);
    }
  }

  console.log('\n3. is any of that actually backed?');
  const inv = Number(ethers.formatUnits(inventory, decimals));
  console.log(`   ${symbol} sitting in all known pools: ${inv.toLocaleString()} ` +
    `(${((inv / supplyN) * 100).toPrecision(2)}% of supply)`);
  if (biggestQuoteTokens > inv) {
    console.log(`   WARNING: the largest quote promised ${biggestQuoteTokens.toFixed(0)} ${symbol}, ` +
      `more than the ${inv.toFixed(0)} that exist in every pool combined.`);
    console.log('   That quote cannot settle. Treat the price above as indicative, not a market.');
  } else if (inv === 0) {
    console.log('   No pool holds any of the token. There is no tradable price.');
  } else {
    console.log('   Quotes are within pool inventory, so there is something behind them.');
  }
} catch (err) {
  console.error(err.shortMessage ?? err.message ?? err);
  process.exitCode = 1;
} finally {
  provider.destroy();
}
