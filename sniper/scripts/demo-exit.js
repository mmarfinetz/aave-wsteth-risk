#!/usr/bin/env node
/**
 * Offline walkthrough of the whole exit path -- plan, place, poll, fill, cancel --
 * against a mock Base node and a mock CoW orderbook. No network, no funds, no real
 * orders. Run it to see what `scripts/exit.js` does before pointing it at mainnet.
 *
 *   npm run demo:exit
 */
import { ethers } from 'ethers';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import { createMockChain } from '../test/mock-chain.js';
import { createMockCow } from '../test/mock-cow.js';

// NOTE: spawn, not spawnSync -- the mock servers live in this process, so blocking the
// event loop would accept the child's TCP connection and then never answer it.
function run(argv, env) {
  return new Promise((resolve) => {
    const child = spawn('node', argv, { env, stdio: ['ignore', 'inherit', 'inherit'] });
    const timer = setTimeout(() => child.kill('SIGTERM'), 30_000);
    child.on('exit', (code) => { clearTimeout(timer); resolve(code); });
  });
}

// Paths below are relative to the package root.
process.chdir(new URL('..', import.meta.url).pathname);

const key = ethers.Wallet.createRandom().privateKey;
const addr = new ethers.Wallet(key).address;

const chain = createMockChain({ walletBalance: ethers.parseEther('1') });
chain.state.tokenBalances[addr.toLowerCase()] = ethers.parseUnits('4900', 18);
const wsUrl = await chain.listenWs();

const cow = createMockCow();
const cowUrl = await cow.listen();

fs.writeFileSync('position.json', JSON.stringify({
  token: '0xB095274743941e953c746F9C228DA9c18Bb6ec29',
  entryWei: ethers.parseEther('0.01').toString(),
  tokensHeld: ethers.parseUnits('4900', 18).toString()
}, null, 2));

const env = {
  ...process.env,
  BASE_WSS: wsUrl, PRIVATE_KEY: key, COW_BASE_URL: cowUrl,
  EXIT_LADDER: '3x:25,5x:25,10x:50', POSITION_FILE: './position.json'
};

for (const mode of ['--plan', '--place', '--status']) {
  console.log(`\n$ node scripts/exit.js ${mode}\n${'-'.repeat(58)}`);
  await run(['scripts/exit.js', mode], env);
}

const placed = JSON.parse(fs.readFileSync('position.orders.json', 'utf8'));
cow.fill(placed[0].uid, {
  sellAmount: ethers.parseUnits('1225', 18), buyAmount: ethers.parseEther('0.00375')
});
console.log(`\n$ node scripts/exit.js --status    # after a solver settles the 3x rung\n${'-'.repeat(58)}`);
await run(['scripts/exit.js', '--status'], env);

console.log(`\n$ node scripts/exit.js --cancel\n${'-'.repeat(58)}`);
await run(['scripts/exit.js', '--cancel'], env);

console.log(`\nsignature rejections by the orderbook: ${cow.rejections.length}`);
console.log(`approved spender(s): ${JSON.stringify(chain.sent.filter(s => s.venue === 'approve').map(s => s.spender))}`);

await chain.closeWs(); await cow.close();
fs.rmSync('position.json', { force: true });
fs.rmSync('position.orders.json', { force: true });
process.exit(0);
