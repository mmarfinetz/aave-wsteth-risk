import test from 'node:test';
import assert from 'node:assert/strict';
import { ethers } from 'ethers';

import * as C from '../src/constants.js';
import { ORDER_TYPES, cowDomain, COW_SETTLEMENT, COW_VAULT_RELAYER } from '../src/cow.js';

/**
 * Interface-identity tests.
 *
 * A function selector is the first four bytes of keccak of the full signature, struct
 * field types and order included. So pinning a selector pins the exact calldata layout
 * the deployed contract expects: if a field is reordered, retyped, added or dropped, the
 * selector changes and this fails. That makes these checks a real assertion about live
 * contract behaviour rather than about a mock -- no network required, and unlike the
 * mocks they cannot drift along with a mistake in src/.
 *
 * Every constant below was cross-checked against an independent source; see
 * VERIFICATION.md for what each was checked against.
 */

const iface = (abi) => new ethers.Interface(abi);
const selector = (abi, name) => iface(abi).getFunction(name).selector;
const sig = (abi, name) => iface(abi).getFunction(name).format('sighash');

// --- addresses -------------------------------------------------------------------

test('Base protocol addresses match the verified deployments', () => {
  assert.equal(C.WETH, '0x4200000000000000000000000000000000000006');
  assert.equal(C.UNI_V3_FACTORY, '0x33128a8fC17869897dcE68Ed026d694621f6FDfD');
  assert.equal(C.UNI_V3_QUOTER, '0x3d4e44Eb1374240CE5F1B871ab261CD16335B76a');
  assert.equal(C.UNI_V3_ROUTER, '0x2626664c2603336E57B271c5C0b26F421741e481');
  assert.equal(C.AERO_FACTORY, '0x420DD381b31aEf6683db6B902084cB0FFECe40Da');
  assert.equal(C.AERO_ROUTER, '0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43');
  assert.equal(C.CHAINLINK_ETH_USD, '0x71041dddad3595F9CEd3DcCFBe3D1F4b0a16Bb70');
  assert.equal(COW_SETTLEMENT, '0x9008D19f58AAbD9eD0D60971565AA8510560ab41');
  assert.equal(COW_VAULT_RELAYER, '0xC92E8bdf79f0507f65a392b0ab4667716BFE0110');
});

test('the target token address is the one that was verified', () => {
  // Changing this is changing what the bot buys. Pinned so it cannot drift silently.
  assert.equal(C.LAPTOP, '0xB095274743941e953c746F9C228DA9c18Bb6ec29');
});

test('every address is checksummed and distinct', () => {
  const all = [
    C.WETH, C.LAPTOP, C.UNI_V3_FACTORY, C.UNI_V3_QUOTER, C.UNI_V3_ROUTER,
    C.AERO_FACTORY, C.AERO_ROUTER, C.CHAINLINK_ETH_USD, COW_SETTLEMENT, COW_VAULT_RELAYER
  ];
  for (const a of all) assert.equal(a, ethers.getAddress(a), `${a} is not checksummed`);
  assert.equal(new Set(all.map((a) => a.toLowerCase())).size, all.length, 'addresses must be distinct');
});

test('CoW allowances go to the relayer, which is not the settlement contract', () => {
  assert.notEqual(COW_VAULT_RELAYER.toLowerCase(), COW_SETTLEMENT.toLowerCase());
});

// --- Uniswap: the V2-vs-V1 distinction the selectors settle -----------------------

test('QuoterV2 selector, not the V1 quoter', () => {
  assert.equal(sig(C.UNI_QUOTER_ABI, 'quoteExactInputSingle'),
    'quoteExactInputSingle((address,address,uint256,uint24,uint160))');
  assert.equal(selector(C.UNI_QUOTER_ABI, 'quoteExactInputSingle'), '0xc6a5026a');
  // V1 took flat args and would encode as 0xf7729d43. Base has only QuoterV2 deployed.
  assert.notEqual(selector(C.UNI_QUOTER_ABI, 'quoteExactInputSingle'), '0xf7729d43');
});

test('QuoterV2 is declared non-view, so callers must use staticCall', () => {
  // QuoterV2 reverts internally and decodes the revert payload; a `view` declaration
  // would let ethers treat it as a plain read and silently misbehave.
  const fn = iface(C.UNI_QUOTER_ABI).getFunction('quoteExactInputSingle');
  assert.equal(fn.constant, false);
  assert.ok(!['view', 'pure'].includes(fn.stateMutability), 'must not be view/pure');
});

test('SwapRouter02 exactInputSingle has no deadline field', () => {
  assert.equal(sig(C.UNI_ROUTER_ABI, 'exactInputSingle'),
    'exactInputSingle((address,address,uint24,address,uint256,uint256,uint160))');
  assert.equal(selector(C.UNI_ROUTER_ABI, 'exactInputSingle'), '0x04e45aaf');
  // The original SwapRouter's struct carried a deadline and encodes as 0x414bf389.
  // Sending that calldata to SwapRouter02 reverts.
  assert.notEqual(selector(C.UNI_ROUTER_ABI, 'exactInputSingle'), '0x414bf389');
});

test('exactInputSingle is payable, so ETH can be wrapped in-flight', () => {
  assert.equal(iface(C.UNI_ROUTER_ABI).getFunction('exactInputSingle').stateMutability, 'payable');
});

test('Uniswap factory getPool selector', () => {
  assert.equal(sig(C.UNI_FACTORY_ABI, 'getPool'), 'getPool(address,address,uint24)');
  assert.equal(selector(C.UNI_FACTORY_ABI, 'getPool'), '0x1698ee82');
});

test('the fee tiers probed are the canonical V3 set', () => {
  assert.deepEqual(C.UNI_FEE_TIERS, [100, 500, 3000, 10000]);
});

// --- Aerodrome -------------------------------------------------------------------

test('Aerodrome factory getPool takes a stable flag, not a fee tier', () => {
  assert.equal(sig(C.AERO_FACTORY_ABI, 'getPool'), 'getPool(address,address,bool)');
  assert.equal(selector(C.AERO_FACTORY_ABI, 'getPool'), '0x79bc57d5');
  // Distinct from the Uniswap factory's same-named function.
  assert.notEqual(selector(C.AERO_FACTORY_ABI, 'getPool'), selector(C.UNI_FACTORY_ABI, 'getPool'));
});

test('Aerodrome route tuple is (from,to,stable,factory)', () => {
  assert.equal(sig(C.AERO_ROUTER_ABI, 'getAmountsOut'),
    'getAmountsOut(uint256,(address,address,bool,address)[])');
  assert.equal(selector(C.AERO_ROUTER_ABI, 'getAmountsOut'), '0x5509a1ac');
  assert.equal(sig(C.AERO_ROUTER_ABI, 'swapExactETHForTokens'),
    'swapExactETHForTokens(uint256,(address,address,bool,address)[],address,uint256)');
  assert.equal(selector(C.AERO_ROUTER_ABI, 'swapExactETHForTokens'), '0x903638a4');
});

test('swapExactETHForTokens is payable', () => {
  assert.equal(iface(C.AERO_ROUTER_ABI).getFunction('swapExactETHForTokens').stateMutability, 'payable');
});

// --- ERC-20 and Chainlink ---------------------------------------------------------

test('ERC-20 selectors are the canonical ones', () => {
  const expected = {
    balanceOf: '0x70a08231', approve: '0x095ea7b3', allowance: '0xdd62ed3e',
    decimals: '0x313ce567', symbol: '0x95d89b41', totalSupply: '0x18160ddd'
  };
  for (const [name, sel] of Object.entries(expected)) {
    assert.equal(selector(C.ERC20_ABI, name), sel, `${name} selector`);
  }
});

test('Chainlink aggregator selectors', () => {
  assert.equal(selector(C.CHAINLINK_ABI, 'latestRoundData'), '0xfeaf968c');
  assert.equal(selector(C.CHAINLINK_ABI, 'decimals'), '0x313ce567');
});

// --- CoW: the signing path --------------------------------------------------------

test('the Order type string matches GPv2Order.sol exactly', () => {
  const encoded = ethers.TypedDataEncoder.from(ORDER_TYPES).encodeType('Order');
  assert.equal(encoded,
    'Order(address sellToken,address buyToken,address receiver,uint256 sellAmount,' +
    'uint256 buyAmount,uint32 validTo,bytes32 appData,uint256 feeAmount,string kind,' +
    'bool partiallyFillable,string sellTokenBalance,string buyTokenBalance)');
});

test('ORDER_TYPE_HASH matches the constant published in GPv2Order.sol', () => {
  // Taken from cowprotocol/contracts src/contracts/libraries/GPv2Order.sol. If a field
  // is reordered this diverges, and solvers reject every order the bot signs.
  const encoded = ethers.TypedDataEncoder.from(ORDER_TYPES).encodeType('Order');
  assert.equal(ethers.keccak256(ethers.toUtf8Bytes(encoded)),
    '0xd5a25ba2e97094ad7d83dc28a6572da797d6b3e7fc6663bd93efb789fc17e489');
});

test('the EIP-712 domain is GPv2 on Base', () => {
  const domain = cowDomain(8453);
  assert.equal(domain.name, 'Gnosis Protocol');
  assert.equal(domain.version, 'v2');
  assert.equal(domain.chainId, 8453);
  assert.equal(domain.verifyingContract, COW_SETTLEMENT);
  // Pinned: the separator binds every signature to this chain and contract.
  assert.equal(ethers.TypedDataEncoder.hashDomain(domain),
    '0xd72ffa789b6fae41254d0b5a13e6e1e92ed947ec6a251edf1cf0b6c02c257b4b');
});

test('the domain separator differs per chain, so a Base signature cannot replay', () => {
  assert.notEqual(
    ethers.TypedDataEncoder.hashDomain(cowDomain(8453)),
    ethers.TypedDataEncoder.hashDomain(cowDomain(1))
  );
});

test('Base chain id is pinned', () => {
  assert.equal(C.BASE_CHAIN_ID, 8453n);
});
