#!/usr/bin/env node
/**
 * Verify every address this bot talks to, by reading it on chain.
 *
 * Goes past "has code": each contract is asked something only it can answer, and where
 * possible contracts are made to vouch for each other -- the quoter names its factory,
 * the router names WETH, the settlement contract names its own vault relayer and domain
 * separator. Cross-references like that are much harder to fake than a label.
 *
 *   BASE_RPC=https://... node scripts/verify-contracts.js
 */
import { ethers } from 'ethers';
import * as C from '../src/constants.js';
import { COW_SETTLEMENT, COW_VAULT_RELAYER, cowDomain } from '../src/cow.js';
import { proxiedProvider, proxyInUse } from './_proxy.js';

const RPC = process.env.BASE_RPC ?? 'https://mainnet.base.org';
const provider = proxiedProvider(RPC);

// A liquid reference pair, for exercising quoting without needing the target token.
const USDC = ethers.getAddress('0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913');

let failures = 0;
const ok = (m) => console.log(`  ok    ${m}`);
const bad = (m) => { failures += 1; console.log(`  FAIL  ${m}`); };
const info = (m) => console.log(`  ..    ${m}`);

const eq = (label, actual, expected) => {
  const same = String(actual).toLowerCase() === String(expected).toLowerCase();
  (same ? ok : bad)(`${label}: ${actual}${same ? '' : ` (expected ${expected})`}`);
  return same;
};

async function hasCode(label, address) {
  const code = await provider.getCode(address);
  if (code === '0x') { bad(`${label} ${address} has NO CODE`); return false; }
  ok(`${label} ${address} has code (${(code.length - 2) / 2} bytes)`);
  return true;
}

const call = async (address, abi, fn, args = []) =>
  new ethers.Contract(address, abi, provider)[fn](...args);

try {
  const net = await provider.getNetwork();
  console.log(`Base chain ${net.chainId}, block ${await provider.getBlockNumber()}`);
  console.log(`RPC: ${RPC.replace(/\/v2\/.*$/, '/v2/<key>')}${proxyInUse ? ' (via proxy)' : ''}\n`);
  if (net.chainId !== 8453n) throw new Error(`not Base: ${net.chainId}`);

  console.log('WETH');
  await hasCode('WETH', C.WETH);
  eq('  symbol', await call(C.WETH, C.ERC20_ABI, 'symbol'), 'WETH');
  eq('  decimals', await call(C.WETH, C.ERC20_ABI, 'decimals'), 18);

  console.log('\nUniswap V3');
  await hasCode('factory', C.UNI_V3_FACTORY);
  const tickSpacing = await call(C.UNI_V3_FACTORY,
    ['function feeAmountTickSpacing(uint24) view returns (int24)'], 'feeAmountTickSpacing', [3000]);
  eq('  feeAmountTickSpacing(3000)', tickSpacing, 60);
  for (const fee of C.UNI_FEE_TIERS) {
    const spacing = await call(C.UNI_V3_FACTORY,
      ['function feeAmountTickSpacing(uint24) view returns (int24)'], 'feeAmountTickSpacing', [fee]);
    if (spacing > 0n) ok(`  fee tier ${fee} is enabled (tick spacing ${spacing})`);
    else bad(`  fee tier ${fee} is NOT enabled on this factory`);
  }

  await hasCode('quoter', C.UNI_V3_QUOTER);
  // The quoter naming our factory ties the two addresses together.
  eq('  QuoterV2.factory()',
    await call(C.UNI_V3_QUOTER, ['function factory() view returns (address)'], 'factory'),
    C.UNI_V3_FACTORY);
  eq('  QuoterV2.WETH9()',
    await call(C.UNI_V3_QUOTER, ['function WETH9() view returns (address)'], 'WETH9'), C.WETH);

  await hasCode('router', C.UNI_V3_ROUTER);
  eq('  SwapRouter02.factory()',
    await call(C.UNI_V3_ROUTER, ['function factory() view returns (address)'], 'factory'),
    C.UNI_V3_FACTORY);
  eq('  SwapRouter02.WETH9()',
    await call(C.UNI_V3_ROUTER, ['function WETH9() view returns (address)'], 'WETH9'), C.WETH);

  // Exercise the exact quoter call the bot makes, on a pair that definitely trades.
  const pool = await call(C.UNI_V3_FACTORY, C.UNI_FACTORY_ABI, 'getPool', [C.WETH, USDC, 500]);
  if (pool === ethers.ZeroAddress) bad('  WETH/USDC 0.05% pool not found');
  else {
    ok(`  WETH/USDC 0.05% pool ${pool}`);
    const quoter = new ethers.Contract(C.UNI_V3_QUOTER, C.UNI_QUOTER_ABI, provider);
    const [out] = await quoter.quoteExactInputSingle.staticCall({
      tokenIn: C.WETH, tokenOut: USDC, amountIn: ethers.parseEther('1'),
      fee: 500, sqrtPriceLimitX96: 0n
    });
    ok(`  live quote decodes: 1 WETH -> ${ethers.formatUnits(out, 6)} USDC`);
  }

  console.log('\nAerodrome');
  await hasCode('factory', C.AERO_FACTORY);
  const poolCount = await call(C.AERO_FACTORY,
    ['function allPoolsLength() view returns (uint256)'], 'allPoolsLength');
  ok(`  factory reports ${poolCount} pools`);
  await hasCode('router', C.AERO_ROUTER);
  eq('  Router.defaultFactory()',
    await call(C.AERO_ROUTER, ['function defaultFactory() view returns (address)'], 'defaultFactory'),
    C.AERO_FACTORY);
  eq('  Router.weth()',
    await call(C.AERO_ROUTER, ['function weth() view returns (address)'], 'weth'), C.WETH);

  console.log('\nChainlink ETH/USD');
  await hasCode('feed', C.CHAINLINK_ETH_USD);
  const desc = await call(C.CHAINLINK_ETH_USD,
    ['function description() view returns (string)'], 'description').catch(() => '(none)');
  eq('  description', desc, 'ETH / USD');
  const feedDecimals = await call(C.CHAINLINK_ETH_USD, C.CHAINLINK_ABI, 'decimals');
  const [, answer, , updatedAt] = await call(C.CHAINLINK_ETH_USD, C.CHAINLINK_ABI, 'latestRoundData');
  const price = Number(ethers.formatUnits(answer, feedDecimals));
  const ageMin = Math.round((Date.now() / 1000 - Number(updatedAt)) / 60);
  if (price > 100 && price < 100_000) ok(`  live ETH/USD = $${price.toFixed(2)} (updated ${ageMin} min ago)`);
  else bad(`  ETH/USD reads ${price}, outside the sane band`);

  console.log('\nCoW Protocol');
  await hasCode('settlement', COW_SETTLEMENT);
  // The settlement contract naming its own relayer verifies the approval target.
  eq('  Settlement.vaultRelayer()',
    await call(COW_SETTLEMENT, ['function vaultRelayer() view returns (address)'], 'vaultRelayer'),
    COW_VAULT_RELAYER);
  const onchainDomain = await call(COW_SETTLEMENT,
    ['function domainSeparator() view returns (bytes32)'], 'domainSeparator');
  const computed = ethers.TypedDataEncoder.hashDomain(cowDomain(8453));
  eq('  Settlement.domainSeparator()', onchainDomain, computed);
  await hasCode('vault relayer', COW_VAULT_RELAYER);

  console.log('\nTarget token');
  const tokenHasCode = await hasCode('LAPTOP', C.LAPTOP);
  if (tokenHasCode) {
    const [sym, dec, supply] = await Promise.all([
      call(C.LAPTOP, C.ERC20_ABI, 'symbol'),
      call(C.LAPTOP, C.ERC20_ABI, 'decimals'),
      call(C.LAPTOP, C.ERC20_ABI, 'totalSupply')
    ]);
    ok(`  symbol ${sym}, decimals ${dec}`);
    ok(`  total supply ${ethers.formatUnits(supply, dec)}`);
    for (const fee of C.UNI_FEE_TIERS) {
      const p = await call(C.UNI_V3_FACTORY, C.UNI_FACTORY_ABI, 'getPool', [C.WETH, C.LAPTOP, fee]);
      if (p === ethers.ZeroAddress) info(`  uni fee ${fee}: no pool yet`);
      else {
        const liq = await call(C.WETH, C.ERC20_ABI, 'balanceOf', [p]);
        ok(`  uni fee ${fee}: pool ${p} holding ${ethers.formatEther(liq)} WETH`);
      }
    }
    const aeroPool = await call(C.AERO_FACTORY, C.AERO_FACTORY_ABI, 'getPool',
      [C.WETH, C.LAPTOP, false]);
    if (aeroPool === ethers.ZeroAddress) info('  aerodrome: no volatile pool yet');
    else {
      const liq = await call(C.WETH, C.ERC20_ABI, 'balanceOf', [aeroPool]);
      ok(`  aerodrome: pool ${aeroPool} holding ${ethers.formatEther(liq)} WETH`);
    }
  }

  console.log(failures === 0
    ? '\nAll on-chain checks passed.'
    : `\n${failures} check(s) FAILED.`);
  process.exitCode = failures === 0 ? 0 : 1;
} catch (err) {
  console.error(`\nERROR: ${err.shortMessage ?? err.message ?? err}`);
  process.exitCode = 1;
} finally {
  provider.destroy();
}
