import test from 'node:test';
import assert from 'node:assert/strict';
import { ethers } from 'ethers';

import { createMockChain } from './mock-chain.js';
import { loadConfig } from '../src/config.js';
import { Watcher } from '../src/watcher.js';
import { WETH } from '../src/constants.js';

const POOL = '0x1111111111111111111111111111111111111111';
const TEST_KEY = ethers.Wallet.createRandom().privateKey;
const quiet = { log() {}, error() {} };

async function waitFor(predicate, { timeout = 12_000, label = 'condition' } = {}) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    if (await predicate()) return true;
    await new Promise((r) => setTimeout(r, 25));
  }
  throw new Error(`timed out waiting for ${label}`);
}

/**
 * Keep producing blocks until the condition holds.
 *
 * A head subscription only delivers blocks produced after eth_subscribe lands, and the
 * chain does not stop for a reconnect -- so a test that mines once up front is racing
 * the subscription. Mining on every poll mirrors what Base actually does.
 */
async function mineUntil(mock, predicate, opts = {}) {
  return waitFor(async () => {
    mock.mineBlock();
    return predicate();
  }, opts);
}

async function startWatcher({ chain = {}, env = {} } = {}) {
  const mock = createMockChain({ walletBalance: ethers.parseEther('1'), ...chain });
  const url = await mock.listenWs();
  const config = loadConfig({
    BASE_WSS: url,
    PRIVATE_KEY: TEST_KEY,
    BUY_ETH: '0.01',
    EARLIEST_BUY: '2020-01-01T00:00:00Z',
    BLOCK_WATCHDOG_MS: '1200',
    ...env
  });
  const watcher = new Watcher({ config, logger: quiet });
  await watcher.start();
  const stop = async () => { await watcher.stop(); await mock.closeWs(); };
  return { mock, watcher, stop };
}

test('scans on every new block delivered over the socket', async () => {
  const h = await startWatcher();
  await mineUntil(h.mock, () => h.watcher.scans >= 2, { label: 'two scans' });
  assert.ok(h.watcher.scans >= 2);
  assert.equal(h.watcher.reconnects, 0);
  await h.stop();
});

test('reconnects when the socket goes deaf without closing', async () => {
  const h = await startWatcher();
  await mineUntil(h.mock, () => h.watcher.scans >= 1, { label: 'first scan' });

  // The socket stays open; heads simply stop arriving. Nothing errors, nothing closes.
  h.mock.goSilent();
  await mineUntil(h.mock, () => h.watcher.reconnects >= 1, { label: 'watchdog to fire' });
  assert.ok(h.watcher.reconnects >= 1, 'watchdog must notice the silence');

  // After the rebuild, block delivery resumes and scanning continues.
  h.mock.resume();
  const before = h.watcher.scans;
  await mineUntil(h.mock, () => h.watcher.scans > before,
    { label: 'scanning to resume after reconnect' });
  assert.ok(h.watcher.scans > before, 'watcher must be live again after reconnect');
  await h.stop();
});

test('a completed buy is not repeated after a reconnect', async () => {
  const h = await startWatcher({
    env: { LIVE: 'true', MIN_TOKENS_OUT: '1' },
    chain: {
      uniPools: { 3000: POOL },
      poolWeth: { [POOL.toLowerCase()]: ethers.parseEther('50') },
      uniQuote: ({ tokenIn }) =>
        tokenIn.toLowerCase() === WETH.toLowerCase()
          ? ethers.parseUnits('5000', 18)
          : ethers.parseEther('0.0098')
    }
  });

  await mineUntil(h.mock, () => h.mock.sent.length >= 1, { label: 'the buy to broadcast' });
  assert.equal(h.mock.sent.length, 1);

  // The watcher stops itself after a fill; further blocks must not produce a second buy.
  h.mock.mineBlock();
  h.mock.mineBlock();
  await new Promise((r) => setTimeout(r, 300));
  assert.equal(h.mock.sent.length, 1, 'exactly one buy, ever');
  await h.stop();
});

test('stop() tears the socket down', async () => {
  const h = await startWatcher();
  await mineUntil(h.mock, () => h.watcher.scans >= 1, { label: 'a scan' });
  await h.watcher.stop();
  await waitFor(() => h.mock.socketCount === 0, { label: 'socket to close' });
  assert.equal(h.mock.socketCount, 0);
  await h.stop();
});
