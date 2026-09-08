import test from 'node:test';
import assert from 'node:assert/strict';
import { ethers } from 'ethers';

import { createMockChain } from './mock-chain.js';
import { loadConfig, slippageMin, maxBigInt } from '../src/config.js';
import { buildContracts } from '../src/wiring.js';
import { Sniper } from '../src/sniper.js';
import { WETH, LAPTOP } from '../src/constants.js';

const POOL_UNI = '0x1111111111111111111111111111111111111111';
const POOL_UNI_ALT = '0x2222222222222222222222222222222222222222';
const POOL_AERO = '0x3333333333333333333333333333333333333333';

// Throwaway key generated for tests. Never funded, never used off this mock chain.
const TEST_KEY = ethers.Wallet.createRandom().privateKey;

const BASE_ENV = {
  BASE_WSS: 'wss://example.invalid',
  PRIVATE_KEY: TEST_KEY,
  BUY_ETH: '0.01',
  MIN_WETH_LIQ: '5',
  SLIPPAGE_BPS: '500',
  EARLIEST_BUY: '2020-01-01T00:00:00Z'
};

const quiet = { log() {}, error() {} };

async function harness({ env = {}, chain = {}, now } = {}) {
  const mock = createMockChain(chain);
  const provider = await mock.listen();
  const config = loadConfig({ ...BASE_ENV, ...env });
  const wallet = new ethers.Wallet(config.PRIVATE_KEY, provider);
  const contracts = buildContracts(provider, wallet);
  const sniper = new Sniper({
    contracts, wallet, config, logger: quiet,
    now: now ?? (() => Date.parse('2026-09-10T00:00:00Z'))
  });
  sniper.setTokenDecimals(18);
  const close = async () => {
    provider.destroy();
    await mock.close();
  };
  return { mock, provider, sniper, config, wallet, close };
}

/** A pool with plenty of WETH that quotes `out` tokens per BUY_ETH. */
function liveUniPool(out, {
  fee = 3000,
  pool = POOL_UNI,
  weth = '50',
  sellBack = ethers.parseEther('0.0098')   // null here means the sell side reverts
} = {}) {
  return {
    uniPools: { [fee]: pool },
    poolWeth: { [pool.toLowerCase()]: ethers.parseEther(weth) },
    uniQuote: ({ tokenIn }) =>
      tokenIn.toLowerCase() === WETH.toLowerCase() ? out : sellBack
  };
}

test('config: rejects LIVE without an absolute price guard', () => {
  assert.throws(
    () => loadConfig({ ...BASE_ENV, LIVE: 'true' }),
    /MIN_TOKENS_OUT/
  );
});

test('config: rejects an out-of-range slippage', () => {
  assert.throws(() => loadConfig({ ...BASE_ENV, SLIPPAGE_BPS: '2500' }), /SLIPPAGE_BPS/);
});

test('config: rejects a non-websocket endpoint', () => {
  assert.throws(() => loadConfig({ ...BASE_ENV, BASE_WSS: 'https://base.example' }), /websocket/);
});

test('config: rejects a malformed EARLIEST_BUY', () => {
  assert.throws(() => loadConfig({ ...BASE_ENV, EARLIEST_BUY: 'next tuesday' }), /EARLIEST_BUY/);
});

test('config: accepts a valid LIVE configuration', () => {
  const cfg = loadConfig({ ...BASE_ENV, LIVE: 'true', MIN_TOKENS_OUT: '1000' });
  assert.equal(cfg.LIVE, true);
  assert.equal(cfg.BUY_WEI, ethers.parseEther('0.01'));
});

test('slippage math: 5% floor, and the absolute guard wins when it is higher', () => {
  assert.equal(slippageMin(1000n, 500n), 950n);
  assert.equal(slippageMin(1000n, 0n), 1000n);
  assert.equal(maxBigInt(slippageMin(1000n, 500n), 990n), 990n);
});

test('does nothing before EARLIEST_BUY, even with liquidity sitting there', async () => {
  const h = await harness({
    env: { EARLIEST_BUY: '2030-01-01T00:00:00Z', LIVE: 'true', MIN_TOKENS_OUT: '1' },
    chain: liveUniPool(ethers.parseUnits('5000', 18))
  });
  const result = await h.sniper.scan(1);
  assert.equal(result.status, 'before-earliest-buy');
  assert.equal(h.mock.sent.length, 0);
  await h.close();
});

test('reports no liquidity when no pool exists', async () => {
  const h = await harness({ env: { LIVE: 'true', MIN_TOKENS_OUT: '1' } });
  const result = await h.sniper.scan(1);
  assert.equal(result.status, 'no-liquidity');
  assert.equal(h.mock.sent.length, 0);
  await h.close();
});

test('skips a pool that is under MIN_WETH_LIQ', async () => {
  const h = await harness({
    env: { LIVE: 'true', MIN_TOKENS_OUT: '1' },
    chain: liveUniPool(ethers.parseUnits('5000', 18), { weth: '1' })  // threshold is 5
  });
  const result = await h.sniper.scan(1);
  assert.equal(result.status, 'no-liquidity');
  const skip = result.diagnostics.find((d) => d.skipped === 'below MIN_WETH_LIQ');
  assert.ok(skip, 'the thin pool should be recorded, not silently dropped');
  assert.equal(h.mock.sent.length, 0);
  await h.close();
});

test('LIVE=false observes and sends nothing', async () => {
  const h = await harness({ chain: liveUniPool(ethers.parseUnits('5000', 18)) });
  const result = await h.sniper.scan(1);
  assert.equal(result.status, 'observed');
  assert.equal(result.op.venue, 'Uniswap V3');
  assert.equal(h.mock.sent.length, 0, 'observation mode must never send');
  await h.close();
});

test('LIVE=true buys, and floors amountOutMinimum at the 5% slippage bound', async () => {
  const quoted = ethers.parseUnits('5000', 18);
  const h = await harness({
    env: { LIVE: 'true', MIN_TOKENS_OUT: '1' },
    chain: liveUniPool(quoted)
  });
  const result = await h.sniper.scan(42);
  assert.equal(result.status, 'bought');
  assert.equal(h.mock.sent.length, 1);
  assert.equal(h.mock.sent[0].venue, 'Uniswap V3');
  assert.equal(h.mock.sent[0].amountOutMin, (quoted * 9500n) / 10_000n);
  assert.equal(h.mock.sent[0].recipient, h.wallet.address);
  assert.equal(h.sniper.bought, true);
  await h.close();
});

test('MIN_TOKENS_OUT overrides slippage when it is the stricter bound', async () => {
  const quoted = ethers.parseUnits('5000', 18);
  const h = await harness({
    env: { LIVE: 'true', MIN_TOKENS_OUT: '4900' },   // above the 4750 slippage floor
    chain: liveUniPool(quoted)
  });
  const result = await h.sniper.scan(1);
  assert.equal(result.status, 'bought');
  assert.equal(h.mock.sent[0].amountOutMin, ethers.parseUnits('4900', 18));
  await h.close();
});

test('picks the venue with the better quote', async () => {
  const h = await harness({
    env: { LIVE: 'true', MIN_TOKENS_OUT: '1' },
    chain: {
      uniPools: { 3000: POOL_UNI },
      aeroPool: POOL_AERO,
      poolWeth: {
        [POOL_UNI.toLowerCase()]: ethers.parseEther('50'),
        [POOL_AERO.toLowerCase()]: ethers.parseEther('50')
      },
      uniQuote: ({ tokenIn }) =>
        tokenIn.toLowerCase() === WETH.toLowerCase()
          ? ethers.parseUnits('4000', 18)
          : ethers.parseEther('0.0098'),
      aeroQuote: ({ from }) =>
        from.toLowerCase() === WETH.toLowerCase()
          ? ethers.parseUnits('6000', 18)   // better
          : ethers.parseEther('0.0098')
    }
  });
  const result = await h.sniper.scan(1);
  assert.equal(result.status, 'bought');
  assert.equal(result.op.venue, 'Aerodrome');
  assert.equal(h.mock.sent[0].venue, 'Aerodrome');
  await h.close();
});

test('picks the best fee tier when several Uniswap pools are live', async () => {
  const h = await harness({
    env: { LIVE: 'true', MIN_TOKENS_OUT: '1' },
    chain: {
      uniPools: { 500: POOL_UNI, 10000: POOL_UNI_ALT },
      poolWeth: {
        [POOL_UNI.toLowerCase()]: ethers.parseEther('50'),
        [POOL_UNI_ALT.toLowerCase()]: ethers.parseEther('50')
      },
      uniQuote: ({ tokenIn, fee }) => {
        if (tokenIn.toLowerCase() !== WETH.toLowerCase()) return ethers.parseEther('0.0098');
        return fee === 10000 ? ethers.parseUnits('7000', 18) : ethers.parseUnits('3000', 18);
      }
    }
  });
  const result = await h.sniper.scan(1);
  assert.equal(result.status, 'bought');
  assert.equal(result.op.fee, 10000);
  assert.equal(h.mock.sent[0].fee, 10000);
  await h.close();
});

test('a reverting fee tier does not sink the whole scan', async () => {
  const h = await harness({
    env: { LIVE: 'true', MIN_TOKENS_OUT: '1' },
    chain: {
      uniPools: { 500: POOL_UNI, 3000: POOL_UNI_ALT },
      poolWeth: {
        [POOL_UNI.toLowerCase()]: ethers.parseEther('50'),
        [POOL_UNI_ALT.toLowerCase()]: ethers.parseEther('50')
      },
      uniQuote: ({ tokenIn, fee }) => {
        if (tokenIn.toLowerCase() !== WETH.toLowerCase()) return ethers.parseEther('0.0098');
        if (fee === 500) return null;   // uninitialized pool -> revert
        return ethers.parseUnits('5000', 18);
      }
    }
  });
  const result = await h.sniper.scan(1);
  assert.equal(result.status, 'bought');
  assert.equal(result.op.fee, 3000);
  await h.close();
});

test('honeypot guard: refuses to buy when the sell side has no route', async () => {
  const h = await harness({
    env: { LIVE: 'true', MIN_TOKENS_OUT: '1' },
    chain: liveUniPool(ethers.parseUnits('5000', 18), { sellBack: null })
  });
  const result = await h.sniper.scan(1);
  assert.equal(result.status, 'blocked');
  assert.equal(result.reason, 'no-sell-path');
  assert.equal(h.mock.sent.length, 0, 'must not buy what it cannot sell');
  await h.close();
});

test('honeypot guard: refuses a round trip that returns almost nothing', async () => {
  const h = await harness({
    env: { LIVE: 'true', MIN_TOKENS_OUT: '1' },
    chain: liveUniPool(ethers.parseUnits('5000', 18), {
      sellBack: ethers.parseEther('0.001')   // 10% back -> 90% sell tax
    })
  });
  const result = await h.sniper.scan(1);
  assert.equal(result.status, 'blocked');
  assert.equal(result.reason, 'low-retention');
  assert.equal(h.mock.sent.length, 0);
  await h.close();
});

test('honeypot guard can be disabled explicitly', async () => {
  const h = await harness({
    env: { LIVE: 'true', MIN_TOKENS_OUT: '1', REQUIRE_SELL_PATH: 'false' },
    chain: liveUniPool(ethers.parseUnits('5000', 18), { sellBack: null })
  });
  const result = await h.sniper.scan(1);
  assert.equal(result.status, 'bought');
  await h.close();
});

test('refuses to size against a quote that went stale mid-scan', async () => {
  let t = Date.parse('2026-09-10T00:00:00Z');
  const h = await harness({
    env: { LIVE: 'true', MIN_TOKENS_OUT: '1', MAX_SCAN_STALENESS_MS: '10' },
    chain: liveUniPool(ethers.parseUnits('5000', 18)),
    now: () => { t += 5_000; return t; }   // clock jumps 5s per read
  });
  const result = await h.sniper.scan(1);
  assert.equal(result.status, 'stale');
  assert.equal(h.mock.sent.length, 0);
  await h.close();
});

test('a reverting router aborts before anything is broadcast', async () => {
  const h = await harness({
    env: { LIVE: 'true', MIN_TOKENS_OUT: '1' },
    chain: { ...liveUniPool(ethers.parseUnits('5000', 18)), uniSwapReverts: true }
  });
  await assert.rejects(() => h.sniper.scan(1), /revert/i);
  assert.equal(h.mock.sent.length, 0, 'staticCall must catch it before the send');
  assert.equal(h.sniper.bought, false);
  await h.close();
});

test('the banner dedupes across blocks instead of reprinting every quote', async () => {
  const lines = [];
  const h = await harness({ chain: liveUniPool(ethers.parseUnits('5000', 18)) });
  h.sniper.logger = { log: (m) => lines.push(String(m)), error() {} };
  await h.sniper.scan(1);
  await h.sniper.scan(2);
  await h.sniper.scan(3);
  const banners = lines.filter((l) => l.includes('Tradable liquidity found'));
  assert.equal(banners.length, 1, 'liquidity banner should print once per pool');
  await h.close();
});
