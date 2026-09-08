import { ethers } from 'ethers';
import { parseLadder } from './exit.js';

const DEFAULTS = {
  BUY_ETH: '0.01',
  MIN_WETH_LIQ: '5',
  SLIPPAGE_BPS: '500',
  EARLIEST_BUY: '2026-09-09T00:00:00-04:00',
  GAS_BUFFER_ETH: '0.0005',
  EXIT_LADDER: '3x:25,5x:25,10x:50',
  EXIT_TTL_HOURS: '72',
  EXIT_RECEIVE: 'WETH',
  COW_BASE_URL: 'https://api.cow.fi',
  POSITION_FILE: './position.json',
  MAX_SCAN_STALENESS_MS: '4000',
  BLOCK_WATCHDOG_MS: '60000'
};

/**
 * Parse and validate configuration.
 *
 * Every check that can be made without the network is made HERE, at startup, so a
 * misconfigured bot dies immediately instead of appearing healthy for hours and then
 * failing at the exact moment liquidity appears.
 */
export function loadConfig(env = process.env) {
  const errors = [];
  const pick = (key) => env[key] ?? DEFAULTS[key];

  const BASE_WSS = env.BASE_WSS;
  if (!BASE_WSS) errors.push('Missing BASE_WSS');
  else if (!/^wss?:\/\//i.test(BASE_WSS)) {
    errors.push('BASE_WSS must be a websocket URL (ws:// or wss://) — block subscriptions need a socket');
  }

  const PRIVATE_KEY = env.PRIVATE_KEY;
  if (!PRIVATE_KEY) errors.push('Missing PRIVATE_KEY');

  const LIVE = String(env.LIVE).toLowerCase() === 'true';

  let BUY_WEI = 0n;
  try {
    BUY_WEI = ethers.parseEther(pick('BUY_ETH'));
    if (BUY_WEI <= 0n) errors.push('BUY_ETH must be greater than 0');
  } catch {
    errors.push(`BUY_ETH is not a valid ether amount: ${pick('BUY_ETH')}`);
  }

  let MIN_WETH_LIQ = 0n;
  try {
    MIN_WETH_LIQ = ethers.parseEther(pick('MIN_WETH_LIQ'));
    if (MIN_WETH_LIQ < 0n) errors.push('MIN_WETH_LIQ cannot be negative');
  } catch {
    errors.push(`MIN_WETH_LIQ is not a valid ether amount: ${pick('MIN_WETH_LIQ')}`);
  }

  let GAS_BUFFER_WEI = 0n;
  try {
    GAS_BUFFER_WEI = ethers.parseEther(pick('GAS_BUFFER_ETH'));
  } catch {
    errors.push(`GAS_BUFFER_ETH is not a valid ether amount: ${pick('GAS_BUFFER_ETH')}`);
  }

  let SLIPPAGE_BPS = 0n;
  try {
    SLIPPAGE_BPS = BigInt(pick('SLIPPAGE_BPS'));
    if (SLIPPAGE_BPS < 0n || SLIPPAGE_BPS > 1500n) {
      errors.push('SLIPPAGE_BPS must be between 0 and 1500 (0%-15%)');
    }
  } catch {
    errors.push(`SLIPPAGE_BPS is not an integer: ${pick('SLIPPAGE_BPS')}`);
  }

  const EARLIEST_BUY_MS = Date.parse(pick('EARLIEST_BUY'));
  if (!Number.isFinite(EARLIEST_BUY_MS)) {
    errors.push(`Invalid EARLIEST_BUY: ${pick('EARLIEST_BUY')}`);
  }

  // The absolute price guard. Slippage alone only protects against movement away from a
  // quote the bot just took -- it does NOT protect against buying into a pool that opened
  // at an absurd price. In LIVE mode this floor is mandatory.
  const MIN_TOKENS_OUT = env.MIN_TOKENS_OUT;
  if (LIVE && !MIN_TOKENS_OUT) {
    errors.push('LIVE=true requires MIN_TOKENS_OUT as an absolute price guard');
  }
  if (MIN_TOKENS_OUT !== undefined && !/^\d+(\.\d+)?$/.test(String(MIN_TOKENS_OUT).trim())) {
    errors.push(`MIN_TOKENS_OUT must be a positive decimal number: ${MIN_TOKENS_OUT}`);
  }

  // Parse the ladder here so a typo fails at startup, not after the position is open.
  let EXIT_LADDER = [];
  try {
    EXIT_LADDER = parseLadder(pick('EXIT_LADDER'));
  } catch (err) {
    errors.push(`EXIT_LADDER: ${err.message}`);
  }

  const EXIT_RECEIVE = String(pick('EXIT_RECEIVE')).toUpperCase();
  if (!['WETH', 'ETH'].includes(EXIT_RECEIVE)) {
    errors.push(`EXIT_RECEIVE must be WETH or ETH, got ${EXIT_RECEIVE}`);
  }

  const EXIT_TTL_HOURS = Number(pick('EXIT_TTL_HOURS'));
  if (!Number.isFinite(EXIT_TTL_HOURS) || EXIT_TTL_HOURS <= 0) {
    errors.push(`EXIT_TTL_HOURS must be a positive number, got ${pick('EXIT_TTL_HOURS')}`);
  }

  const REQUIRE_SELL_PATH = String(env.REQUIRE_SELL_PATH ?? 'true').toLowerCase() === 'true';
  const MAX_SCAN_STALENESS_MS = Number(pick('MAX_SCAN_STALENESS_MS'));
  const BLOCK_WATCHDOG_MS = Number(pick('BLOCK_WATCHDOG_MS'));

  if (errors.length > 0) {
    const err = new Error(`Invalid configuration:\n  - ${errors.join('\n  - ')}`);
    err.errors = errors;
    throw err;
  }

  return {
    BASE_WSS,
    PRIVATE_KEY,
    LIVE,
    BUY_WEI,
    MIN_WETH_LIQ,
    GAS_BUFFER_WEI,
    SLIPPAGE_BPS,
    EARLIEST_BUY_MS,
    MIN_TOKENS_OUT,
    REQUIRE_SELL_PATH,
    MAX_SCAN_STALENESS_MS,
    BLOCK_WATCHDOG_MS,
    EXIT_LADDER,
    EXIT_LADDER_SPEC: pick('EXIT_LADDER'),
    EXIT_TTL_HOURS,
    EXIT_RECEIVE,
    COW_BASE_URL: pick('COW_BASE_URL'),
    POSITION_FILE: pick('POSITION_FILE')
  };
}

export const maxBigInt = (a, b) => (a > b ? a : b);

export const slippageMin = (quotedOut, slippageBps) =>
  (quotedOut * (10_000n - slippageBps)) / 10_000n;
