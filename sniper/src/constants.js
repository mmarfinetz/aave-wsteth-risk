import { ethers } from 'ethers';

// Official $LAPTOP token on Base. Do not change unless the official project changes it.
// NOTE: this address is asserted, not verified by this repo. Re-verify it against the
// project's own announcement before any live run — see README "Verify the token".
export const LAPTOP = ethers.getAddress('0xB095274743941e953c746F9C228DA9c18Bb6ec29');
export const WETH = ethers.getAddress('0x4200000000000000000000000000000000000006');

// Uniswap V3 — Base
export const UNI_V3_FACTORY = ethers.getAddress('0x33128a8fC17869897dcE68Ed026d694621f6FDfD');
export const UNI_V3_QUOTER = ethers.getAddress('0x3d4e44Eb1374240CE5F1B871ab261CD16335B76a');
export const UNI_V3_ROUTER = ethers.getAddress('0x2626664c2603336E57B271c5C0b26F421741e481');
export const UNI_FEE_TIERS = [100, 500, 3000, 10000];

// Aerodrome classic pools — Base
export const AERO_FACTORY = ethers.getAddress('0x420DD381b31aEf6683db6B902084cB0FFECe40Da');
export const AERO_ROUTER = ethers.getAddress('0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43');

export const BASE_CHAIN_ID = 8453n;
export const ZERO = ethers.ZeroAddress;

export const ERC20_ABI = [
  'function symbol() view returns (string)',
  'function decimals() view returns (uint8)',
  'function totalSupply() view returns (uint256)',
  'function balanceOf(address) view returns (uint256)',
  'function allowance(address owner,address spender) view returns (uint256)',
  'function approve(address spender,uint256 amount) returns (bool)'
];

export const UNI_FACTORY_ABI = [
  'function getPool(address tokenA,address tokenB,uint24 fee) view returns (address pool)'
];

// QuoterV2 is intentionally non-view: it reverts internally and decodes the revert data,
// so every call must go through staticCall.
export const UNI_QUOTER_ABI = [
  'function quoteExactInputSingle((address tokenIn,address tokenOut,uint256 amountIn,uint24 fee,uint160 sqrtPriceLimitX96) params) returns (uint256 amountOut,uint160 sqrtPriceX96After,uint32 initializedTicksCrossed,uint256 gasEstimate)'
];

// SwapRouter02 — note there is no deadline field on this struct.
export const UNI_ROUTER_ABI = [
  'function exactInputSingle((address tokenIn,address tokenOut,uint24 fee,address recipient,uint256 amountIn,uint256 amountOutMinimum,uint160 sqrtPriceLimitX96) params) payable returns (uint256 amountOut)'
];

export const AERO_FACTORY_ABI = [
  'function getPool(address tokenA,address tokenB,bool stable) view returns (address pool)'
];

export const AERO_ROUTER_ABI = [
  'function getAmountsOut(uint256 amountIn,(address from,address to,bool stable,address factory)[] routes) view returns (uint256[] amounts)',
  'function swapExactETHForTokens(uint256 amountOutMin,(address from,address to,bool stable,address factory)[] routes,address to,uint256 deadline) payable returns (uint256[] amounts)',
  'function swapExactTokensForETH(uint256 amountIn,uint256 amountOutMin,(address from,address to,bool stable,address factory)[] routes,address to,uint256 deadline) returns (uint256[] amounts)'
];
