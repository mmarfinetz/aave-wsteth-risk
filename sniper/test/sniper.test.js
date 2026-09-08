import test from 'node:test';
import assert from 'node:assert/strict';
import { ethers } from 'ethers';

import { createMockChain, cpQuote } from './mock-chain.js';
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

/**
 * A live Uniswap pool, priced off real reserves so size costs what it should.
 * `sellBack` overrides only the sell direction, for the honeypot cases.
 */
function liveUniPool({
  fee = 3000,
  pool = POOL_UNI,
  weth = '50',
  tokens = '25000000',
  sellBack
} = {}) {
  const key = pool.toLowerCase();
  const reserves = {
    weth: ethers.parseEther(weth),
    token: ethers.parseUnits(tokens, 18)
  };
  const chain = {
    uniPools: { [fee]: pool },
    poolWeth: { [key]: reserves.weth },
    poolReserves: { [key]: reserves }
  };

  if (sellBack !== undefined) {
    chain.uniQuote = ({ tokenIn, amountIn }) =>
      tokenIn.toLowerCase() === WETH.toLowerCase()
        ? cpQuote({ amountIn, reserveIn: reserves.weth, reserveOut: reserves.token })
        : sellBack;
  }
  return chain;
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
    chain: liveUniPool()
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
    chain: liveUniPool({ weth: '1' })   // threshold is 5
  });
  const result = await h.sniper.scan(1);
  assert.equal(result.status, 'no-liquidity');
  const skip = result.diagnostics.find((d) => d.skipped === 'below MIN_WETH_LIQ');
  assert.ok(skip, 'the thin pool should be recorded, not silently dropped');
  assert.equal(h.mock.sent.length, 0);
  await h.close();
});

test('LIVE=false observes and sends nothing', async () => {
  const h = await harness({ chain: liveUniPool() });
  const result = await h.sniper.scan(1);
  assert.equal(result.status, 'observed');
  assert.equal(result.op.venue, 'Uniswap V3');
  assert.equal(h.mock.sent.length, 0, 'observation mode must never send');
  await h.close();
});

test('LIVE=true buys, and floors amountOutMinimum at the 5% slippage bound', async () => {
  const h = await harness({
    env: { LIVE: 'true', MIN_TOKENS_OUT: '1' },
    chain: liveUniPool()
  });
  const result = await h.sniper.scan(42);
  assert.equal(result.status, 'bought');
  assert.equal(h.mock.sent.length, 1);
  assert.equal(h.mock.sent[0].venue, 'Uniswap V3');
  assert.equal(h.mock.sent[0].amountOutMin, (result.op.quotedOut * 9500n) / 10_000n);
  assert.equal(h.mock.sent[0].recipient, h.wallet.address);
  assert.equal(h.sniper.bought, true);
  await h.close();
});

test('MIN_TOKENS_OUT overrides slippage when it is the stricter bound', async () => {
  // The pool quotes ~4984 tokens for 0.01 ETH, so a 4900 floor sits above the
  // 5% slippage bound (~4735) and must win.
  const h = await harness({
    env: { LIVE: 'true', MIN_TOKENS_OUT: '4900' },
    chain: liveUniPool()
  });
  const result = await h.sniper.scan(1);
  assert.equal(result.status, 'bought');
  const slippageFloor = (result.op.quotedOut * 9500n) / 10_000n;
  assert.ok(ethers.parseUnits('4900', 18) > slippageFloor, 'fixture must exercise the override');
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
      poolReserves: {
        [POOL_UNI.toLowerCase()]:
          { weth: ethers.parseEther('50'), token: ethers.parseUnits('20000000', 18) },
        // Same WETH depth, more token per ETH -> the better quote.
        [POOL_AERO.toLowerCase()]:
          { weth: ethers.parseEther('50'), token: ethers.parseUnits('30000000', 18) }
      }
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
      poolReserves: {
        [POOL_UNI.toLowerCase()]:
          { weth: ethers.parseEther('50'), token: ethers.parseUnits('10000000', 18) },
        [POOL_UNI_ALT.toLowerCase()]:
          { weth: ethers.parseEther('50'), token: ethers.parseUnits('35000000', 18) }
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
      poolReserves: {
        [POOL_UNI_ALT.toLowerCase()]:
          { weth: ethers.parseEther('50'), token: ethers.parseUnits('25000000', 18) }
      },
      uniQuote: ({ tokenIn, amountIn, fee }) => {
        if (fee === 500) return null;   // uninitialized pool -> revert
        if (tokenIn.toLowerCase() !== WETH.toLowerCase()) return ethers.parseEther('0.0098');
        return cpQuote({
          amountIn,
          reserveIn: ethers.parseEther('50'),
          reserveOut: ethers.parseUnits('25000000', 18)
        });
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
    chain: liveUniPool({ sellBack: null })
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
    chain: liveUniPool({ sellBack: ethers.parseEther('0.001') })   // 10% back
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
    chain: liveUniPool({ sellBack: null })
  });
  const result = await h.sniper.scan(1);
  assert.equal(result.status, 'bought');
  await h.close();
});

test('refuses to size against a quote that went stale mid-scan', async () => {
  let t = Date.parse('2026-09-10T00:00:00Z');
  const h = await harness({
    env: { LIVE: 'true', MIN_TOKENS_OUT: '1', MAX_SCAN_STALENESS_MS: '10' },
    chain: liveUniPool(),
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
    chain: { ...liveUniPool(), uniSwapReverts: true }
  });
  await assert.rejects(() => h.sniper.scan(1), /revert/i);
  assert.equal(h.mock.sent.length, 0, 'staticCall must catch it before the send');
  assert.equal(h.sniper.bought, false);
  await h.close();
});

test('a fill reports the position the exit ladder needs to price against', async () => {
  const filled = ethers.parseUnits('4900', 18);   // less than quoted, e.g. a transfer tax
  const h = await harness({
    env: { LIVE: 'true', MIN_TOKENS_OUT: '1' },
    chain: liveUniPool()
  });
  // The wallet ends up holding what actually landed, not what was quoted.
  h.mock.state.tokenBalances[h.wallet.address.toLowerCase()] = filled;

  const result = await h.sniper.scan(7);
  assert.equal(result.status, 'bought');
  assert.equal(result.tokensHeld, filled, 'must report the real balance, not the quote');
  assert.equal(result.entryWei, ethers.parseEther('0.01'));
  assert.equal(result.blockNumber, 30_000_000);
  assert.ok(result.txHash?.startsWith('0x'));
  await h.close();
});

// --- sizing: what a four-figure buy does to a launch pool ------------------------

test('blocks a $1k-sized buy against a pool too thin to absorb it', async () => {
  // $1000 at $4,400 is ~0.227 ETH. Into a 1 WETH pool that is ~18% price impact.
  const h = await harness({
    env: {
      LIVE: 'true', MIN_TOKENS_OUT: '1',
      BUY_ETH: '0.227', MIN_WETH_LIQ: '0.5', MAX_PRICE_IMPACT_BPS: '1000'
    },
    chain: liveUniPool({ weth: '1', tokens: '500000' })
  });
  const result = await h.sniper.scan(1);
  assert.equal(result.status, 'blocked');
  assert.equal(result.reason, 'price-impact');
  assert.ok(result.impact > 1000n, `impact should exceed the cap, got ${result.impact}`);
  assert.equal(h.mock.sent.length, 0, 'must not buy into a pool this thin');
  await h.close();
});

test('allows the same buy once the pool is deep enough', async () => {
  const h = await harness({
    env: {
      LIVE: 'true', MIN_TOKENS_OUT: '1',
      BUY_ETH: '0.227', MIN_WETH_LIQ: '5', MAX_PRICE_IMPACT_BPS: '1000'
    },
    chain: liveUniPool({ weth: '200', tokens: '100000000' })
  });
  const result = await h.sniper.scan(1);
  assert.equal(result.status, 'bought');
  assert.ok(result.op.priceImpactBps < 1000n,
    `deep pool should be low impact, got ${result.op.priceImpactBps}`);
  await h.close();
});

test('the impact cap can be disabled with 0', async () => {
  const h = await harness({
    env: {
      LIVE: 'true', MIN_TOKENS_OUT: '1',
      BUY_ETH: '0.227', MIN_WETH_LIQ: '0.5', MAX_PRICE_IMPACT_BPS: '0'
    },
    chain: liveUniPool({ weth: '1', tokens: '500000' })
  });
  const result = await h.sniper.scan(1);
  assert.equal(result.status, 'bought');
  await h.close();
});

test('the banner dedupes across blocks instead of reprinting every quote', async () => {
  const lines = [];
  const h = await harness({ chain: liveUniPool() });
  h.sniper.logger = { log: (m) => lines.push(String(m)), error() {} };
  await h.sniper.scan(1);
  await h.sniper.scan(2);
  await h.sniper.scan(3);
  const banners = lines.filter((l) => l.includes('Tradable liquidity found'));
  assert.equal(banners.length, 1, 'liquidity banner should print once per pool');
  await h.close();
});
