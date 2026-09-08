#!/usr/bin/env node
/**
 * Place, inspect and cancel the CoW take-profit ladder.
 *
 *   node scripts/exit.js --plan      show the rungs, touch nothing
 *   node scripts/exit.js --place     approve if needed, rest every rung on CoW
 *   node scripts/exit.js --status    poll fills
 *   node scripts/exit.js --cancel    cancel every resting rung
 *
 * Entry price comes from position.json (written when the sniper fills) or from
 * ENTRY_ETH + the live token balance.
 */
import 'dotenv/config';
import fs from 'node:fs';
import { ethers } from 'ethers';
import { loadConfig } from '../src/config.js';
import { buildContracts } from '../src/wiring.js';
import { ExitManager, buildRungs } from '../src/exit.js';
import { CowClient, ETH_SENTINEL } from '../src/cow.js';
import { LAPTOP, WETH } from '../src/constants.js';

const args = new Set(process.argv.slice(2));
const mode = ['--plan', '--place', '--status', '--cancel'].find((m) => args.has(m)) ?? '--plan';

const config = loadConfig();
const provider = new ethers.WebSocketProvider(config.BASE_WSS);
const wallet = new ethers.Wallet(config.PRIVATE_KEY, provider);
const contracts = buildContracts(provider, wallet);

const ordersFile = config.POSITION_FILE.replace(/\.json$/, '') + '.orders.json';

function readJson(path, fallback = null) {
  try { return JSON.parse(fs.readFileSync(path, 'utf8')); } catch { return fallback; }
}

async function resolvePosition() {
  const recorded = readJson(config.POSITION_FILE);
  const held = await contracts.laptop.balanceOf(wallet.address);

  const entryWei = recorded?.entryWei
    ? BigInt(recorded.entryWei)
    : (process.env.ENTRY_ETH ? ethers.parseEther(process.env.ENTRY_ETH) : null);

  if (!entryWei) {
    throw new Error(
      `No entry price. Either ${config.POSITION_FILE} is missing (the sniper writes it ` +
      `on a fill) or ENTRY_ETH is unset.`
    );
  }
  if (held <= 0n) throw new Error('Wallet holds none of the token; nothing to exit.');

  // Ladder against what is actually held now, not what was bought -- a transfer tax or
  // a partial earlier exit would otherwise price every rung wrong.
  return { totalTokens: held, entryWei, recorded };
}

const manager = new ExitManager({
  client: new CowClient({ chainId: 8453, baseUrl: config.COW_BASE_URL }),
  wallet,
  token: contracts.laptopWrite,
  sellToken: LAPTOP,
  buyToken: config.EXIT_RECEIVE === 'ETH' ? ETH_SENTINEL : WETH,
  ttlSeconds: Math.round(config.EXIT_TTL_HOURS * 3600)
});

try {
  if (mode === '--status') {
    manager.placed = readJson(ordersFile, []);
    if (manager.placed.length === 0) {
      console.log(`No recorded orders in ${ordersFile}.`);
    } else {
      for (const row of await manager.status()) {
        console.log(`${String(row.rung).padEnd(10)} ${row.status ?? row.error}` +
          (row.executedSellAmount && row.executedSellAmount !== '0'
            ? `  filled ${ethers.formatUnits(row.executedSellAmount, 18)} tokens ` +
              `for ${ethers.formatEther(row.executedBuyAmount)} ETH`
            : ''));
      }
    }
  } else if (mode === '--cancel') {
    manager.placed = readJson(ordersFile, []);
    if (manager.placed.length === 0) throw new Error(`No recorded orders in ${ordersFile}`);
    await manager.cancelAll();
    console.log(`Cancelled ${manager.placed.length} rung(s).`);
  } else {
    const { totalTokens, entryWei } = await resolvePosition();
    const rungs = buildRungs({ totalTokens, entryWei, ladder: config.EXIT_LADDER });

    console.log(`position:  ${ethers.formatUnits(totalTokens, 18)} tokens`);
    console.log(`entry:     ${ethers.formatEther(entryWei)} ETH`);
    console.log(`receive:   ${config.EXIT_RECEIVE}`);
    console.log(`ladder:    ${config.EXIT_LADDER_SPEC}\n`);

    let proceeds = 0n;
    for (const r of rungs) {
      proceeds += r.buyAmount;
      console.log(`  ${r.label.padEnd(9)} sell ${ethers.formatUnits(r.sellAmount, 18).padStart(14)}` +
        ` for ${ethers.formatEther(r.buyAmount).padStart(12)} ETH`);
    }
    console.log(`\n  if every rung fills: ${ethers.formatEther(proceeds)} ETH ` +
      `(${(Number(proceeds * 100n / entryWei) / 100).toFixed(2)}x on entry)`);

    if (mode === '--plan') {
      console.log('\n--plan only. Nothing approved, nothing placed.');
    } else {
      console.log('\nPlacing...');
      const result = await manager.placeLadder({
        totalTokens, entryWei, ladder: config.EXIT_LADDER
      });
      fs.writeFileSync(ordersFile, JSON.stringify(result.placed, null, 2));
      for (const p of result.placed) console.log(`  ${p.rung.padEnd(9)} ${p.uid}`);
      console.log(`\nRecorded ${result.placed.length} rung(s) in ${ordersFile}`);
    }
  }
} catch (err) {
  console.error(err.message ?? err);
  provider.destroy();
  process.exit(1);
}

provider.destroy();
process.exit(0);
