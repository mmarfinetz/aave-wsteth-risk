import test from 'node:test';
import assert from 'node:assert/strict';
import { ethers } from 'ethers';

import { createMockChain } from './mock-chain.js';
import { createMockCow } from './mock-cow.js';
import { buildContracts } from '../src/wiring.js';
import { parseLadder, buildRungs, resolveAppData, ExitManager } from '../src/exit.js';
import { CowClient, COW_VAULT_RELAYER, COW_SETTLEMENT, computeOrderUid, ORDER_TYPES, cowDomain } from '../src/cow.js';
import { LAPTOP, WETH } from '../src/constants.js';

const quiet = { log() {}, error() {} };
const TOTAL = ethers.parseUnits('5000', 18);
const ENTRY = ethers.parseEther('0.01');

async function harness({ cow = {}, chain = {} } = {}) {
  const mockChain = createMockChain({
    walletBalance: ethers.parseEther('1'),
    ...chain
  });
  const provider = await mockChain.listen();
  const wallet = ethers.Wallet.createRandom().connect(provider);
  const contracts = buildContracts(provider, wallet);
  const mockCow = createMockCow(cow);
  const baseUrl = await mockCow.listen();

  const manager = new ExitManager({
    client: new CowClient({ chainId: 8453, baseUrl }),
    wallet,
    token: contracts.laptopWrite,
    sellToken: LAPTOP,
    buyToken: WETH,
    logger: quiet
  });

  const close = async () => { provider.destroy(); await mockChain.close(); await mockCow.close(); };
  return { mockChain, mockCow, manager, wallet, close };
}

// --- ladder parsing -------------------------------------------------------------

test('parses a ladder spec', () => {
  const ladder = parseLadder('3x:25,5x:25,10x:50');
  assert.equal(ladder.length, 3);
  assert.equal(ladder[0].multipleBps, 30_000n);
  assert.equal(ladder[0].pctBps, 2_500n);
});

test('rejects a ladder selling more than the position', () => {
  assert.throws(() => parseLadder('2x:60,4x:60'), /cannot exceed 100%/);
});

test('rejects malformed rungs', () => {
  assert.throws(() => parseLadder('3x'), /not <multiple>x:<percent>/);
  assert.throws(() => parseLadder('threex:25'), /not <multiple>x:<percent>/);
  assert.throws(() => parseLadder(''), /empty/);
});

test('accepts fractional multiples', () => {
  const ladder = parseLadder('1.5x:50');
  assert.equal(ladder[0].multipleBps, 15_000n);
});

// --- rung math ------------------------------------------------------------------

test('prices each rung at its multiple of the entry price', () => {
  const rungs = buildRungs({ totalTokens: TOTAL, entryWei: ENTRY, ladder: parseLadder('3x:25,5x:25,10x:50') });

  assert.equal(rungs[0].sellAmount, ethers.parseUnits('1250', 18));
  assert.equal(rungs[0].buyAmount, ethers.parseEther('0.0075'));   // 25% at 3x
  assert.equal(rungs[1].buyAmount, ethers.parseEther('0.0125'));   // 25% at 5x
  assert.equal(rungs[2].buyAmount, ethers.parseEther('0.05'));     // 50% at 10x

  const sold = rungs.reduce((s, r) => s + r.sellAmount, 0n);
  assert.equal(sold, TOTAL, 'a full ladder must sell the whole position');
});

test('the final rung sweeps the rounding remainder', () => {
  // 3 equal rungs over an amount not divisible by 3.
  const odd = ethers.parseUnits('5000', 18) + 1n;
  const rungs = buildRungs({ totalTokens: odd, entryWei: ENTRY, ladder: parseLadder('2x:33.33,3x:33.33,4x:33.34') });
  const sold = rungs.reduce((s, r) => s + r.sellAmount, 0n);
  assert.equal(sold, odd, 'no dust may be stranded');
});

test('a partial ladder leaves the remainder unsold', () => {
  const rungs = buildRungs({ totalTokens: TOTAL, entryWei: ENTRY, ladder: parseLadder('3x:25') });
  assert.equal(rungs.reduce((s, r) => s + r.sellAmount, 0n), TOTAL / 4n);
});

// --- appData shapes -------------------------------------------------------------

test('resolveAppData handles a bare hash and a doc-plus-hash', () => {
  const hash = '0x' + '11'.repeat(32);
  assert.deepEqual(resolveAppData({ quote: { appData: hash } }), { signed: hash, posted: hash });

  const doc = '{"version":"0.9.0"}';
  const resolved = resolveAppData({ quote: {}, appData: doc, appDataHash: hash });
  assert.equal(resolved.signed, hash);
  assert.equal(resolved.posted, doc);

  assert.equal(resolveAppData({}).signed, ethers.ZeroHash);
});

// --- placement ------------------------------------------------------------------

test('places every rung with a signature the orderbook can recover', async () => {
  const h = await harness();
  const result = await h.manager.placeLadder({
    totalTokens: TOTAL, entryWei: ENTRY, ladder: parseLadder('3x:25,5x:25,10x:50')
  });

  assert.equal(result.placed.length, 3);
  assert.equal(h.mockCow.rejections.length, 0, 'no signature may be rejected');
  assert.equal(h.mockCow.orders.size, 3);

  // Every stored order must carry our wallet as the recovered signer.
  for (const record of result.placed) {
    const stored = h.mockCow.orders.get(record.uid.toLowerCase());
    assert.ok(stored, `order ${record.rung} should be on the book`);
    assert.equal(stored.from.toLowerCase(), h.wallet.address.toLowerCase());
    assert.equal(record.uid.toLowerCase(), record.expectedUid.toLowerCase(),
      'locally computed UID must match the orderbook');
  }
  await h.close();
});

test('approves the vault relayer, never the settlement contract', async () => {
  const h = await harness();
  await h.manager.placeLadder({
    totalTokens: TOTAL, entryWei: ENTRY, ladder: parseLadder('3x:100')
  });

  const approvals = h.mockChain.sent.filter((s) => s.venue === 'approve');
  assert.equal(approvals.length, 1);
  assert.equal(approvals[0].spender, COW_VAULT_RELAYER);
  assert.notEqual(approvals[0].spender, COW_SETTLEMENT);
  await h.close();
});

test('skips the approval when allowance is already sufficient', async () => {
  const mockChain = createMockChain({ walletBalance: ethers.parseEther('1') });
  const provider = await mockChain.listen();
  const wallet = ethers.Wallet.createRandom().connect(provider);
  const contracts = buildContracts(provider, wallet);
  mockChain.state.allowances[`${wallet.address.toLowerCase()}:${COW_VAULT_RELAYER.toLowerCase()}`] =
    ethers.MaxUint256;

  const mockCow = createMockCow();
  const baseUrl = await mockCow.listen();
  const manager = new ExitManager({
    client: new CowClient({ chainId: 8453, baseUrl }),
    wallet, token: contracts.laptopWrite,
    sellToken: LAPTOP, buyToken: WETH, logger: quiet
  });

  await manager.placeLadder({ totalTokens: TOTAL, entryWei: ENTRY, ladder: parseLadder('3x:100') });
  assert.equal(mockChain.sent.filter((s) => s.venue === 'approve').length, 0);

  provider.destroy(); await mockChain.close(); await mockCow.close();
});

test('refuses to place anything when no solver can route the token', async () => {
  const h = await harness({ cow: { noRoute: true } });
  await assert.rejects(
    () => h.manager.placeLadder({ totalTokens: TOTAL, entryWei: ENTRY, ladder: parseLadder('3x:100') }),
    /cannot route this token/
  );
  // Crucially: no approval was broadcast before discovering this.
  assert.equal(h.mockChain.sent.filter((s) => s.venue === 'approve').length, 0);
  assert.equal(h.mockCow.orders.size, 0);
  await h.close();
});

test('a tampered order is rejected by signature recovery', async () => {
  const h = await harness();
  const client = h.manager.client;
  const order = h.manager.buildOrder({
    sellAmount: 1000n, buyAmount: 2000n, appData: ethers.ZeroHash
  });
  const signature = await h.wallet.signTypedData(cowDomain(8453), ORDER_TYPES, order);

  // Raise buyAmount after signing: the recovered signer no longer matches.
  await assert.rejects(
    () => client.placeOrder({
      ...order, buyAmount: '999999', signingScheme: 'eip712',
      signature, from: h.wallet.address
    }),
    /InvalidSignature|recovered/
  );
  assert.equal(h.mockCow.rejections.length, 1);
  assert.equal(h.mockCow.rejections[0].reason, 'signer-mismatch');
  await h.close();
});

// --- lifecycle ------------------------------------------------------------------

test('reports fills, including partial ones', async () => {
  const h = await harness();
  const result = await h.manager.placeLadder({
    totalTokens: TOTAL, entryWei: ENTRY, ladder: parseLadder('3x:50,5x:50')
  });

  h.mockCow.fill(result.placed[0].uid, {
    sellAmount: ethers.parseUnits('1250', 18),
    buyAmount: ethers.parseEther('0.00375'),
    partial: true
  });
  h.mockCow.fill(result.placed[1].uid, {});

  const status = await h.manager.status();
  assert.equal(status[0].status, 'open');
  assert.equal(status[0].executedSellAmount, ethers.parseUnits('1250', 18).toString());
  assert.equal(status[1].status, 'fulfilled');
  await h.close();
});

test('cancels every resting rung', async () => {
  const h = await harness();
  await h.manager.placeLadder({
    totalTokens: TOTAL, entryWei: ENTRY, ladder: parseLadder('3x:50,5x:50')
  });
  await h.manager.cancelAll();
  const status = await h.manager.status();
  assert.ok(status.every((s) => s.status === 'cancelled'), 'all rungs cancelled');
  await h.close();
});

test('surfaces a CoW API error with its status and body', async () => {
  const h = await harness();
  await assert.rejects(() => h.manager.client.getOrder('0xdeadbeef'), (err) => {
    assert.equal(err.name, 'CowError');
    assert.equal(err.status, 404);
    return true;
  });
  await h.close();
});
