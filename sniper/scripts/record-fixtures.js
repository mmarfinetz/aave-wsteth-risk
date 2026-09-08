#!/usr/bin/env node
/**
 * Record real Base responses so tests can replay actual bytes instead of invented ones.
 *
 * The mocks under test/ encode this repo's assumptions about how Base contracts behave,
 * so where an assumption is wrong the mock and the code are wrong together and the suite
 * still passes. A recorded fixture has no such failure mode: the bytes came off Base.
 *
 *   BASE_RPC=https://... TOKEN=0x<a token with liquidity> node scripts/record-fixtures.js
 *
 * Writes test/fixtures/base-<token>.json. Commit it; the replay test picks it up
 * automatically and CI then exercises real wire data with no network of its own.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { ethers } from 'ethers';
import * as C from '../src/constants.js';
import { proxiedRequest } from './_proxy.js';

const RPC = process.env.BASE_RPC;
const TOKEN = process.env.TOKEN ?? C.LAPTOP;
const PROBE_ETH = process.env.PROBE_ETH ?? '0.05';

if (!RPC) {
  console.error('Set BASE_RPC to a Base mainnet endpoint (https:// is fine).');
  process.exit(1);
}

/** Records every eth_call that passes through, with its raw result. */
class RecordingProvider extends ethers.JsonRpcProvider {
  constructor(url) {
    super(proxiedRequest(url), { chainId: 8453, name: 'base' }, { staticNetwork: true });
    this.recorded = [];
  }
  async call(tx) {
    const entry = { to: ethers.getAddress(tx.to), data: ethers.hexlify(tx.data) };
    try {
      entry.result = await super.call(tx);
      this.recorded.push(entry);
      return entry.result;
    } catch (err) {
      // A revert is real behaviour -- an uninitialized pool reverts on quote, and the
      // bot has to handle that. Recording only successes would replay a chain where
      // every call works, which is the opposite of what a launch looks like.
      entry.revert = true;
      entry.errorData = err.data ?? '0x';
      entry.errorMessage = err.shortMessage ?? err.message ?? 'execution reverted';
      this.recorded.push(entry);
      throw err;
    }
  }
}

const provider = new RecordingProvider(RPC);
const token = ethers.getAddress(TOKEN);
const amountIn = ethers.parseEther(PROBE_ETH);

const erc20 = new ethers.Contract(token, C.ERC20_ABI, provider);
const weth = new ethers.Contract(C.WETH, C.ERC20_ABI, provider);
const uniFactory = new ethers.Contract(C.UNI_V3_FACTORY, C.UNI_FACTORY_ABI, provider);
const uniQuoter = new ethers.Contract(C.UNI_V3_QUOTER, C.UNI_QUOTER_ABI, provider);
const aeroFactory = new ethers.Contract(C.AERO_FACTORY, C.AERO_FACTORY_ABI, provider);
const aeroRouter = new ethers.Contract(C.AERO_ROUTER, C.AERO_ROUTER_ABI, provider);
const feed = new ethers.Contract(C.CHAINLINK_ETH_USD, C.CHAINLINK_ABI, provider);

const note = (m) => console.log(`  ${m}`);
const meta = {};

try {
  const network = await provider.getNetwork();
  if (network.chainId !== C.BASE_CHAIN_ID) {
    throw new Error(`Expected Base 8453, got ${network.chainId}`);
  }
  meta.blockNumber = await provider.getBlockNumber();
  meta.chainId = Number(network.chainId);
  console.log(`Recording against Base at block ${meta.blockNumber}\n`);

  meta.code = (await provider.getCode(token)) !== '0x';
  note(`token has code: ${meta.code}`);

  const [symbol, decimals, supply] = await Promise.all([
    erc20.symbol(), erc20.decimals(), erc20.totalSupply()
  ]);
  meta.symbol = symbol;
  meta.decimals = Number(decimals);
  meta.totalSupply = supply.toString();
  note(`${symbol}, ${decimals} decimals, supply ${ethers.formatUnits(supply, decimals)}`);

  const feedDecimals = Number(await feed.decimals());
  const [, answer, , updatedAt] = await feed.latestRoundData();
  meta.ethUsd = { answer: answer.toString(), decimals: feedDecimals, updatedAt: Number(updatedAt) };
  note(`ETH/USD ${ethers.formatUnits(answer, feedDecimals)} (updated ${new Date(Number(updatedAt) * 1000).toISOString()})`);

  meta.uniPools = {};
  for (const fee of C.UNI_FEE_TIERS) {
    const pool = await uniFactory.getPool(C.WETH, token, fee);
    meta.uniPools[fee] = pool;
    if (pool === ethers.ZeroAddress) { note(`uni fee ${fee}: no pool`); continue; }
    const liq = await weth.balanceOf(pool);
    note(`uni fee ${fee}: ${pool} holding ${ethers.formatEther(liq)} WETH`);
    try {
      const [out] = await uniQuoter.quoteExactInputSingle.staticCall({
        tokenIn: C.WETH, tokenOut: token, amountIn, fee, sqrtPriceLimitX96: 0n
      });
      note(`  quotes ${ethers.formatUnits(out, decimals)} for ${PROBE_ETH} ETH`);
      // Record the sell direction too, so the honeypot probe replays as well.
      await uniQuoter.quoteExactInputSingle.staticCall({
        tokenIn: token, tokenOut: C.WETH, amountIn: out, fee, sqrtPriceLimitX96: 0n
      });
      // And a small probe, for the price-impact estimate.
      await uniQuoter.quoteExactInputSingle.staticCall({
        tokenIn: C.WETH, tokenOut: token, amountIn: amountIn / 100n, fee, sqrtPriceLimitX96: 0n
      });
    } catch (err) {
      note(`  quote reverted: ${err.shortMessage ?? err.message}`);
    }
  }

  const aeroPool = await aeroFactory.getPool(C.WETH, token, false);
  meta.aeroPool = aeroPool;
  if (aeroPool !== ethers.ZeroAddress) {
    const liq = await weth.balanceOf(aeroPool);
    note(`aerodrome: ${aeroPool} holding ${ethers.formatEther(liq)} WETH`);
    const routes = [{ from: C.WETH, to: token, stable: false, factory: C.AERO_FACTORY }];
    try {
      const amounts = await aeroRouter.getAmountsOut(amountIn, routes);
      note(`  quotes ${ethers.formatUnits(amounts.at(-1), decimals)} for ${PROBE_ETH} ETH`);
      await aeroRouter.getAmountsOut(amountIn / 100n, routes);
      await aeroRouter.getAmountsOut(amounts.at(-1),
        [{ from: token, to: C.WETH, stable: false, factory: C.AERO_FACTORY }]);
    } catch (err) {
      note(`  quote reverted: ${err.shortMessage ?? err.message}`);
    }
  } else {
    note('aerodrome: no volatile pool');
  }

  const dir = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'test', 'fixtures');
  fs.mkdirSync(dir, { recursive: true });
  const file = path.join(dir, `base-${token.toLowerCase()}.json`);
  fs.writeFileSync(file, JSON.stringify({
    recordedAt: new Date().toISOString(),
    token, probeEth: PROBE_ETH, meta,
    calls: provider.recorded
  }, null, 2));

  console.log(`\nRecorded ${provider.recorded.length} eth_call responses to ${file}`);
  console.log('Commit it and `npm test` will replay real Base bytes.');
} catch (err) {
  console.error(err.shortMessage ?? err.message ?? err);
  process.exitCode = 1;
} finally {
  provider.destroy();
}
