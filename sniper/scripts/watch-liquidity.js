#!/usr/bin/env node
/**
 * Poll Base until a WETH pool for the token crosses a depth threshold, then exit.
 *
 * Exit codes are the signal, so whatever supervises this knows what happened without
 * parsing output: 0 = threshold crossed, 2 = time budget spent, 3 = too many consecutive
 * RPC failures.
 *
 *   BASE_RPC=https://... THRESHOLD_WETH=5 node scripts/watch-liquidity.js
 */
import { ethers } from 'ethers';
import fs from 'node:fs';
import { proxiedProvider } from './_proxy.js';
import { buildContracts } from '../src/wiring.js';
import { LAPTOP, WETH, UNI_FEE_TIERS } from '../src/constants.js';

const RPC = process.env.BASE_RPC ?? 'https://mainnet.base.org';
const TOKEN = ethers.getAddress(process.env.TOKEN ?? LAPTOP);
const THRESHOLD = ethers.parseEther(process.env.THRESHOLD_WETH ?? '5');
const POLL_SEC = Number(process.env.POLL_SEC ?? '60');
const MAX_HOURS = Number(process.env.MAX_HOURS ?? '12');
const LOG = process.env.WATCH_LOG ?? './liquidity-watch.log';

const provider = proxiedProvider(RPC);
const wallet = ethers.Wallet.createRandom().connect(provider);
const contracts = buildContracts(provider, wallet);

const deadline = Date.now() + MAX_HOURS * 3600_000;
const stamp = () => new Date().toISOString().replace('T', ' ').slice(0, 19);
const line = (m) => {
  const s = `${stamp()}  ${m}`;
  console.log(s);
  try { fs.appendFileSync(LOG, s + '\n'); } catch { /* log is best effort */ }
};

/** Current WETH depth of every pool that exists, plus the deepest. */
async function poll() {
  const pools = [];
  for (const fee of UNI_FEE_TIERS) {
    const pool = await contracts.uniFactory.getPool(WETH, TOKEN, fee);
    if (pool === ethers.ZeroAddress) continue;
    pools.push({ venue: `uni${fee}`, pool, weth: await contracts.weth.balanceOf(pool) });
  }
  const aero = await contracts.aeroFactory.getPool(WETH, TOKEN, false);
  if (aero !== ethers.ZeroAddress) {
    pools.push({ venue: 'aero', pool: aero, weth: await contracts.weth.balanceOf(aero) });
  }
  const deepest = pools.reduce((a, b) => (b.weth > (a?.weth ?? -1n) ? b : a), null);
  return { pools, deepest };
}

line(`watch start: ${TOKEN} threshold ${ethers.formatEther(THRESHOLD)} WETH, ` +
     `poll ${POLL_SEC}s, budget ${MAX_HOURS}h`);

let consecutiveFailures = 0;
let lastSignature = '';

while (Date.now() < deadline) {
  try {
    const { pools, deepest } = await poll();
    consecutiveFailures = 0;

    // Only log when something actually changed, so the log stays readable over hours.
    const signature = pools.map((p) => `${p.venue}:${p.weth}`).join('|');
    if (signature !== lastSignature) {
      lastSignature = signature;
      line(pools.length === 0
        ? 'no pools yet'
        : pools.map((p) => `${p.venue} ${Number(ethers.formatEther(p.weth)).toFixed(4)}`).join('  '));
    }

    if (deepest && deepest.weth >= THRESHOLD) {
      line(`THRESHOLD CROSSED: ${deepest.venue} ${deepest.pool} holds ` +
           `${ethers.formatEther(deepest.weth)} WETH (>= ${ethers.formatEther(THRESHOLD)})`);
      provider.destroy();
      process.exit(0);
    }
  } catch (err) {
    consecutiveFailures += 1;
    line(`poll failed (${consecutiveFailures}): ${err.shortMessage ?? err.message}`);
    if (consecutiveFailures >= 10) {
      line('giving up after 10 consecutive failures');
      provider.destroy();
      process.exit(3);
    }
  }
  await new Promise((r) => setTimeout(r, POLL_SEC * 1000));
}

line(`time budget of ${MAX_HOURS}h spent without crossing the threshold`);
provider.destroy();
process.exit(2);
