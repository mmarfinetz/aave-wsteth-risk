import test from 'node:test';
import assert from 'node:assert/strict';
import { ethers } from 'ethers';

import { createMockChain } from './mock-chain.js';
import { loadConfig } from '../src/config.js';
import { buildContracts } from '../src/wiring.js';
import { preflight } from '../src/preflight.js';

const TEST_KEY = ethers.Wallet.createRandom().privateKey;

const BASE_ENV = {
  BASE_WSS: 'wss://example.invalid',
  PRIVATE_KEY: TEST_KEY,
  BUY_ETH: '0.01',
  EARLIEST_BUY: '2020-01-01T00:00:00Z'
};

async function run({ env = {}, chain = {} } = {}) {
  const mock = createMockChain(chain);
  const provider = await mock.listen();
  const config = loadConfig({ ...BASE_ENV, ...env });
  const wallet = new ethers.Wallet(config.PRIVATE_KEY, provider);
  const contracts = buildContracts(provider, wallet);
  const close = async () => { provider.destroy(); await mock.close(); };
  return { provider, wallet, contracts, config, close };
}

test('passes on Base with a funded wallet and a real token', async () => {
  const h = await run({ chain: { walletBalance: ethers.parseEther('0.5') } });
  const info = await preflight(h);
  assert.equal(info.symbol, 'LAPTOP');
  assert.equal(Number(info.decimals), 18);
  assert.equal(info.balance, ethers.parseEther('0.5'));
  await h.close();
});

test('refuses to arm on the wrong chain', async () => {
  const h = await run({ chain: { chainId: 1 } });   // Ethereum mainnet, not Base
  await assert.rejects(() => preflight(h), /Wrong chain/);
  await h.close();
});

test('refuses to arm when the token address has no code', async () => {
  const h = await run({ chain: { tokenHasCode: false } });
  await assert.rejects(() => preflight(h), /no code/);
  await h.close();
});

test('refuses a wallet funded with exactly BUY_ETH and nothing for gas', async () => {
  const h = await run({
    env: { BUY_ETH: '0.01', GAS_BUFFER_ETH: '0.0005' },
    chain: { walletBalance: ethers.parseEther('0.01') }
  });
  await assert.rejects(() => preflight(h), /needs at least/);
  await h.close();
});

test('accepts a wallet holding BUY_ETH plus the gas buffer', async () => {
  const h = await run({
    env: { BUY_ETH: '0.01', GAS_BUFFER_ETH: '0.0005' },
    chain: { walletBalance: ethers.parseEther('0.0105') }
  });
  const info = await preflight(h);
  assert.equal(info.balance, ethers.parseEther('0.0105'));
  await h.close();
});
