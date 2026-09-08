import test from 'node:test';
import assert from 'node:assert/strict';
import { ethers } from 'ethers';

import { createMockChain } from './mock-chain.js';
import { buildContracts } from '../src/wiring.js';
import { readEthUsd, usdToWei, priceImpactBps } from '../src/pricing.js';
import { loadConfig } from '../src/config.js';
import { preflight } from '../src/preflight.js';

async function feedHarness(chain = {}) {
  const mock = createMockChain({ walletBalance: ethers.parseEther('1'), ...chain });
  const provider = await mock.listen();
  const wallet = ethers.Wallet.createRandom().connect(provider);
  const contracts = buildContracts(provider, wallet);
  const close = async () => { provider.destroy(); await mock.close(); };
  return { mock, provider, wallet, contracts, close };
}

test('reads ETH/USD and converts a USD notional to wei', async () => {
  const h = await feedHarness();
  const price = await readEthUsd(h.contracts.ethUsdFeed);
  assert.equal(price.usd, 4400n);

  const wei = usdToWei(1000, price);
  // $1000 / $4400 = 0.22727... ETH
  assert.equal(ethers.formatEther(wei).slice(0, 8), '0.227272');
  await h.close();
});

test('refuses a stale price rather than sizing a trade off it', async () => {
  const h = await feedHarness({ ethUsdUpdatedAt: Math.floor(Date.now() / 1000) - 7200 });
  await assert.rejects(() => readEthUsd(h.contracts.ethUsdFeed, { maxAgeSec: 3600 }), /stale/);
  await h.close();
});

test('refuses a price outside the sanity band', async () => {
  // A stablecoin feed pointed at by mistake reads ~1.00.
  const h = await feedHarness({ ethUsdPrice: 100000000n });
  await assert.rejects(() => readEthUsd(h.contracts.ethUsdFeed), /outside the sane band/);
  await h.close();
});

test('refuses a non-positive price', async () => {
  const h = await feedHarness({ ethUsdPrice: 0n });
  await assert.rejects(() => readEthUsd(h.contracts.ethUsdFeed), /non-positive/);
  await h.close();
});

test('preflight turns BUY_USD into a concrete wei amount', async () => {
  const h = await feedHarness({ walletBalance: ethers.parseEther('1') });
  const config = loadConfig({
    BASE_WSS: 'wss://x', PRIVATE_KEY: h.wallet.privateKey,
    BUY_USD: '1000', EARLIEST_BUY: '2020-01-01T00:00:00Z'
  });
  assert.equal(config.BUY_WEI, 0n, 'unresolved until the feed is read');

  const info = await preflight({ provider: h.provider, wallet: h.wallet, contracts: h.contracts, config });
  assert.equal(info.ethUsd.usd, 4400n);
  assert.equal(ethers.formatEther(config.BUY_WEI).slice(0, 8), '0.227272');
  await h.close();
});

test('preflight rejects a USD buy the wallet cannot cover', async () => {
  const h = await feedHarness({ walletBalance: ethers.parseEther('0.1') });   // < 0.227
  const config = loadConfig({
    BASE_WSS: 'wss://x', PRIVATE_KEY: h.wallet.privateKey,
    BUY_USD: '1000', EARLIEST_BUY: '2020-01-01T00:00:00Z'
  });
  await assert.rejects(
    () => preflight({ provider: h.provider, wallet: h.wallet, contracts: h.contracts, config }),
    /needs at least/
  );
  await h.close();
});

test('price impact rises with size against the same pool', () => {
  // Rates taken from a 50 WETH / 25M token pool.
  const small = priceImpactBps({
    probeIn: 10n ** 14n, probeOut: 49_849_900_599_298_204_999n,
    actualIn: 10n ** 16n, actualOut: 4_984_006_189_165_880_323_463n
  });
  const large = priceImpactBps({
    probeIn: 10n ** 14n, probeOut: 49_849_900_599_298_204_999n,
    actualIn: 227n * 10n ** 15n, actualOut: 112_649_605_080_555_475_307_676n
  });
  assert.ok(large > small, 'a bigger trade must show more impact');
  assert.ok(small < 50n, `a tiny trade should be near zero impact, got ${small}`);
});

test('price impact is zero when the rate does not degrade', () => {
  assert.equal(priceImpactBps({
    probeIn: 100n, probeOut: 1000n, actualIn: 10_000n, actualOut: 100_000n
  }), 0n);
});

test('price impact is null when a probe is unavailable', () => {
  assert.equal(priceImpactBps({ probeIn: 0n, probeOut: 0n, actualIn: 1n, actualOut: 1n }), null);
});
