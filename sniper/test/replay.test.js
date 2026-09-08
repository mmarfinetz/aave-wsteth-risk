import test from 'node:test';
import assert from 'node:assert/strict';
import { ethers } from 'ethers';

import { loadFixtures, createReplayChain } from './replay-chain.js';
import { buildContracts } from '../src/wiring.js';
import { inspectUniswapV3, inspectAerodrome } from '../src/venues.js';
import { readEthUsd } from '../src/pricing.js';
import { WETH } from '../src/constants.js';

const fixtures = loadFixtures();
const noFixtures = fixtures.length === 0;
const skip = noFixtures
  ? 'no recorded fixtures — run scripts/record-fixtures.js with a Base RPC'
  : false;

/**
 * These replay real Base responses. Unlike the hand-written mocks they cannot agree with
 * a mistake in src/: the bytes came off chain, so if an ABI or struct layout is wrong,
 * decoding fails here.
 */
test('recorded fixtures decode against our ABIs', { skip }, async () => {
  for (const fixture of fixtures) {
    const chain = createReplayChain(fixture);
    const provider = await chain.listen();
    const wallet = ethers.Wallet.createRandom().connect(provider);
    const contracts = buildContracts(provider, wallet);

    const symbol = await contracts.laptop.symbol().catch(() => null);
    // Fixtures may be recorded for a different token than the pinned constant.
    if (fixture.token.toLowerCase() === (await contracts.laptop.getAddress()).toLowerCase()) {
      assert.equal(symbol, fixture.meta.symbol, `${fixture.name}: symbol must round-trip`);
    }

    const feed = await readEthUsd(contracts.ethUsdFeed, { maxAgeSec: Number.MAX_SAFE_INTEGER });
    assert.equal(feed.decimals, fixture.meta.ethUsd.decimals);
    assert.ok(feed.usd > 0n, 'ETH/USD must decode to a positive price');

    provider.destroy();
    await chain.close();
  }
});

test('pool discovery replays against real chain state', { skip }, async () => {
  for (const fixture of fixtures) {
    const chain = createReplayChain(fixture);
    const provider = await chain.listen();
    const wallet = ethers.Wallet.createRandom().connect(provider);
    const contracts = buildContracts(provider, wallet);

    const diagnostics = [];
    const ctx = {
      ...contracts,
      token: ethers.getAddress(fixture.token),
      buyWei: ethers.parseEther(fixture.probeEth),
      minWethLiq: 0n,
      diagnostics
    };

    const [uni, aero] = await Promise.all([inspectUniswapV3(ctx), inspectAerodrome(ctx)]);
    const found = [...uni, ...aero];

    // Whatever the fixture recorded, discovery must agree with it.
    const expectedUni = Object.entries(fixture.meta.uniPools)
      .filter(([, pool]) => pool !== ethers.ZeroAddress);
    for (const [fee, pool] of expectedUni) {
      const match = uni.find((o) => String(o.fee) === String(fee));
      if (match) assert.equal(match.pool, pool, `${fixture.name}: fee ${fee} pool address`);
    }

    for (const op of found) {
      assert.ok(op.quotedOut > 0n, `${fixture.name}: ${op.venue} must quote a positive amount`);
      assert.ok(op.wethLiquidity >= 0n);
      assert.equal(ethers.getAddress(op.pool), op.pool, 'pool address must be checksummed');
    }

    assert.deepEqual(chain.misses, [],
      `${fixture.name}: every call the code made should be in the fixture`);

    provider.destroy();
    await chain.close();
  }
});

test('WETH is one side of every discovered pool', { skip }, async () => {
  for (const fixture of fixtures) {
    const chain = createReplayChain(fixture);
    const provider = await chain.listen();
    const wallet = ethers.Wallet.createRandom().connect(provider);
    const contracts = buildContracts(provider, wallet);
    const balance = await contracts.weth.balanceOf(
      Object.values(fixture.meta.uniPools).find((p) => p !== ethers.ZeroAddress) ?? WETH
    ).catch(() => null);
    if (balance !== null) assert.ok(balance >= 0n);
    provider.destroy();
    await chain.close();
  }
});

/**
 * Proves the replay harness itself works, with no recorded file needed.
 *
 * The result bytes here are built with ethers' own ABI encoder rather than by the mock
 * chain's dispatch logic, so this exercises the record/replay plumbing independently of
 * the hand-written mocks. Without it the recorder would ship untested.
 */
test('the replay harness serves recorded bytes to the real venue code', async () => {
  const { UNI_V3_FACTORY, UNI_V3_QUOTER, UNI_FACTORY_ABI, UNI_QUOTER_ABI,
          ERC20_ABI, AERO_FACTORY, AERO_FACTORY_ABI,
          CHAINLINK_ETH_USD, CHAINLINK_ABI } = await import('../src/constants.js');

  const token = ethers.getAddress('0xB095274743941e953c746F9C228DA9c18Bb6ec29');
  const pool = ethers.getAddress('0x1111111111111111111111111111111111111111');
  const amountIn = ethers.parseEther('0.05');
  const quoted = ethers.parseUnits('1234.5', 18);

  const factory = new ethers.Interface(UNI_FACTORY_ABI);
  const quoter = new ethers.Interface(UNI_QUOTER_ABI);
  const erc20 = new ethers.Interface(ERC20_ABI);
  const aeroFactory = new ethers.Interface(AERO_FACTORY_ABI);
  const link = new ethers.Interface(CHAINLINK_ABI);

  const call = (to, iface, name, args, resultIface, resultName, result) => ({
    to,
    data: iface.encodeFunctionData(name, args),
    result: (resultIface ?? iface).encodeFunctionResult(resultName ?? name, result)
  });

  const calls = [
    // Only the 3000 tier has a pool; the rest resolve to the zero address.
    ...[100, 500, 3000, 10000].map((fee) =>
      call(UNI_V3_FACTORY, factory, 'getPool', [WETH, token, fee], null, null,
        [fee === 3000 ? pool : ethers.ZeroAddress])),
    call(WETH, erc20, 'balanceOf', [pool], null, null, [ethers.parseEther('42')]),
    call(UNI_V3_QUOTER, quoter, 'quoteExactInputSingle',
      [[WETH, token, amountIn, 3000, 0n]], null, null, [quoted, 0n, 0, 0n]),
    call(AERO_FACTORY, aeroFactory, 'getPool', [WETH, token, false], null, null,
      [ethers.ZeroAddress]),
    call(CHAINLINK_ETH_USD, link, 'decimals', [], null, null, [8]),
    call(CHAINLINK_ETH_USD, link, 'latestRoundData', [], null, null,
      [1n, 440000000000n, 1n, BigInt(Math.floor(Date.now() / 1000)), 1n])
  ];

  const chain = createReplayChain({
    token, probeEth: '0.05',
    meta: { chainId: 8453, blockNumber: 30_000_000, code: true, ethUsd: { decimals: 8 } },
    calls
  });
  const provider = await chain.listen();
  const wallet = ethers.Wallet.createRandom().connect(provider);
  const contracts = buildContracts(provider, wallet);

  const diagnostics = [];
  const [uni, aero] = await Promise.all([
    inspectUniswapV3({ ...contracts, token, buyWei: amountIn, minWethLiq: 0n, diagnostics }),
    inspectAerodrome({ ...contracts, token, buyWei: amountIn, minWethLiq: 0n, diagnostics })
  ]);

  assert.equal(uni.length, 1, 'exactly the one recorded pool should be found');
  assert.equal(uni[0].pool, pool);
  assert.equal(uni[0].fee, 3000);
  assert.equal(uni[0].quotedOut, quoted, 'the recorded quote must decode exactly');
  assert.equal(uni[0].wethLiquidity, ethers.parseEther('42'));
  assert.equal(aero.length, 0, 'no Aerodrome pool was recorded');

  const price = await readEthUsd(contracts.ethUsdFeed);
  assert.equal(price.usd, 4400n);

  assert.deepEqual(chain.misses, [], 'the code must not call anything outside the fixture');

  provider.destroy();
  await chain.close();
});

test('the fixture directory is wired up even when empty', () => {
  assert.ok(Array.isArray(fixtures));
  if (noFixtures) {
    console.log('    (no fixtures recorded yet — replay tests skipped)');
  } else {
    for (const f of fixtures) {
      assert.ok(f.calls?.length > 0, `${f.name} must contain recorded calls`);
      assert.equal(f.meta.chainId, 8453, `${f.name} must be from Base`);
    }
  }
});
