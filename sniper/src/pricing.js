import { ethers } from 'ethers';

// A wrong feed address, or a feed for the wrong pair, usually reads far outside this
// band -- a stablecoin feed returns ~1, a broken one returns 0. Sizing a four-figure
// trade off either is worse than not trading.
export const MIN_SANE_ETH_USD = 100n;
export const MAX_SANE_ETH_USD = 100_000n;

/** Default tolerance for a stale oracle round, in seconds. */
export const DEFAULT_MAX_FEED_AGE_SEC = 3600;

/**
 * Read ETH/USD and convert a USD notional into wei.
 *
 * Sizing in USD is what people actually mean by "a thousand dollars of it", but it puts
 * an oracle in the path of a trade, so the read is checked three ways: the round must be
 * fresh, the price must be positive, and it must land inside a sanity band.
 */
export async function readEthUsd(feed, { now = () => Date.now(), maxAgeSec = DEFAULT_MAX_FEED_AGE_SEC } = {}) {
  const [, answer, , updatedAt] = await feed.latestRoundData();
  const decimals = Number(await feed.decimals());

  if (answer <= 0n) throw new Error(`ETH/USD feed returned a non-positive price: ${answer}`);

  const ageSec = Math.floor(now() / 1000) - Number(updatedAt);
  if (ageSec > maxAgeSec) {
    throw new Error(
      `ETH/USD feed is stale: last update ${ageSec}s ago (limit ${maxAgeSec}s). ` +
      'Refusing to size a trade off it.'
    );
  }

  const scale = 10n ** BigInt(decimals);
  const whole = answer / scale;
  if (whole < MIN_SANE_ETH_USD || whole > MAX_SANE_ETH_USD) {
    throw new Error(
      `ETH/USD feed reads $${whole}, outside the sane band ` +
      `$${MIN_SANE_ETH_USD}-$${MAX_SANE_ETH_USD}. Check the feed address.`
    );
  }

  return { price: answer, decimals, scale, usd: whole, ageSec };
}

/** USD notional -> wei, at the given ETH/USD price. */
export function usdToWei(usdAmount, { price, scale }) {
  const usdScaled = ethers.parseUnits(String(usdAmount), 18);
  return (usdScaled * scale) / price;
}

/**
 * Estimate price impact by comparing the real order against a probe 1/100th its size.
 *
 * The probe has impact of its own, so this understates slightly -- it is a guard against
 * buying into a pool far too thin for the size, not a precise figure.
 */
export function priceImpactBps({ probeIn, probeOut, actualIn, actualOut }) {
  if (probeIn <= 0n || probeOut <= 0n || actualIn <= 0n || actualOut <= 0n) return null;
  // Tokens per wei, scaled to keep integer division honest.
  const SCALE = 10n ** 18n;
  const referenceRate = (probeOut * SCALE) / probeIn;
  const actualRate = (actualOut * SCALE) / actualIn;
  if (referenceRate === 0n) return null;
  if (actualRate >= referenceRate) return 0n;
  return ((referenceRate - actualRate) * 10_000n) / referenceRate;
}
