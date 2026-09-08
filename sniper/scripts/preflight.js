#!/usr/bin/env node
/**
 * Dry-run checker. Connects to real Base, validates config, wallet, token and pool
 * state, and prints what the watcher would do -- without ever sending a transaction.
 * Run this before the launch window, and again an hour before it.
 *
 *   npm run preflight
 */
import 'dotenv/config';
import { ethers } from 'ethers';
import { loadConfig } from '../src/config.js';
import { buildContracts } from '../src/wiring.js';
import { preflight } from '../src/preflight.js';
import { inspectUniswapV3, inspectAerodrome } from '../src/venues.js';
import { LAPTOP, UNI_FEE_TIERS } from '../src/constants.js';

const ok = (m) => console.log(`  ok    ${m}`);
const bad = (m) => console.log(`  FAIL  ${m}`);
const info = (m) => console.log(`  ..    ${m}`);

const config = loadConfig();
console.log('Config parsed and validated.');
console.log(`  LIVE=${config.LIVE}  BUY_ETH=${ethers.formatEther(config.BUY_WEI)}  ` +
  `MIN_WETH_LIQ=${ethers.formatEther(config.MIN_WETH_LIQ)}  ` +
  `SLIPPAGE=${Number(config.SLIPPAGE_BPS) / 100}%`);
console.log(`  EARLIEST_BUY=${new Date(config.EARLIEST_BUY_MS).toISOString()}`);
console.log(`  MIN_TOKENS_OUT=${config.MIN_TOKENS_OUT ?? '(unset)'}`);
console.log(`  sell-path gate=${config.REQUIRE_SELL_PATH}`);

const provider = new ethers.WebSocketProvider(config.BASE_WSS);
const wallet = new ethers.Wallet(config.PRIVATE_KEY, provider);
const contracts = buildContracts(provider, wallet);

let failed = false;

try {
  console.log('\nConnectivity and wallet');
  const meta = await preflight({ provider, wallet, contracts, config });
  ok(`chain is Base 8453, socket alive`);
  ok(`token ${LAPTOP} responds: ${meta.symbol}, ${meta.decimals} decimals`);
  ok(`wallet ${wallet.address} holds ${ethers.formatEther(meta.balance)} ETH`);
  info(`total supply ${ethers.formatUnits(meta.supply, meta.decimals)}`);

  const block = await provider.getBlockNumber();
  ok(`head block ${block}`);

  console.log('\nPool discovery (read-only)');
  const diagnostics = [];
  const ctx = {
    ...contracts,
    buyWei: config.BUY_WEI,
    minWethLiq: config.MIN_WETH_LIQ,
    diagnostics
  };
  const [uni, aero] = await Promise.all([inspectUniswapV3(ctx), inspectAerodrome(ctx)]);
  const found = [...uni, ...aero];

  if (found.length === 0) {
    info('no tradable pool clears MIN_WETH_LIQ yet (expected before launch)');
    for (const d of diagnostics) {
      const label = d.fee ? `${d.venue} ${d.fee}` : d.venue;
      if (d.error) info(`${label}: error - ${d.error}`);
      else if (d.wethLiquidity !== undefined) {
        info(`${label}: pool exists, ${ethers.formatEther(d.wethLiquidity)} WETH - ${d.skipped}`);
      } else info(`${label}: ${d.skipped}`);
    }
  } else {
    for (const op of found) {
      ok(`${op.venue}${op.fee ? ` fee ${op.fee}` : ''} pool ${op.pool} ` +
         `holds ${ethers.formatEther(op.wethLiquidity)} WETH, quotes ` +
         `${ethers.formatUnits(op.quotedOut, meta.decimals)} tokens for ` +
         `${ethers.formatEther(config.BUY_WEI)} ETH`);
    }
  }

  console.log('\nTiming');
  const until = config.EARLIEST_BUY_MS - Date.now();
  if (until > 0) info(`${Math.round(until / 60_000)} minutes until EARLIEST_BUY`);
  else ok('EARLIEST_BUY has passed; the watcher would be free to buy');

  console.log(`\nUniswap fee tiers probed: ${UNI_FEE_TIERS.join(', ')}`);
  console.log(failed ? '\nPreflight FAILED.' : '\nPreflight passed. No transaction was sent.');
} catch (err) {
  failed = true;
  bad(err.shortMessage ?? err.message ?? String(err));
  console.log('\nPreflight FAILED.');
} finally {
  provider.destroy();
  process.exit(failed ? 1 : 0);
}
