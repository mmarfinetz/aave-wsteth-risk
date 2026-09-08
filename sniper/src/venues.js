import { WETH, ZERO, UNI_FEE_TIERS, AERO_FACTORY } from './constants.js';

/**
 * Probe every Uniswap V3 fee tier in parallel.
 *
 * Parallel matters: the original sequential version made up to 8 round trips before it
 * could send, which on a 2s-block chain meant the quote it sized against was often a
 * block stale by the time the swap landed.
 */
export async function inspectUniswapV3(ctx) {
  const { uniFactory, weth, uniQuoter, buyWei, minWethLiq, diagnostics, token } = ctx;

  const probes = UNI_FEE_TIERS.map(async (fee) => {
    const pool = await uniFactory.getPool(WETH, token, fee);
    if (!pool || pool === ZERO) {
      diagnostics.push({ venue: 'Uniswap V3', fee, skipped: 'no pool' });
      return null;
    }

    const wethLiquidity = await weth.balanceOf(pool);
    if (wethLiquidity < minWethLiq) {
      diagnostics.push({
        venue: 'Uniswap V3', fee, pool,
        skipped: 'below MIN_WETH_LIQ',
        wethLiquidity
      });
      return null;
    }

    const [quotedOut] = await uniQuoter.quoteExactInputSingle.staticCall({
      tokenIn: WETH,
      tokenOut: token,
      amountIn: buyWei,
      fee,
      sqrtPriceLimitX96: 0n
    });

    if (quotedOut <= 0n) {
      diagnostics.push({ venue: 'Uniswap V3', fee, pool, skipped: 'zero quote' });
      return null;
    }

    return { venue: 'Uniswap V3', pool, fee, quotedOut, wethLiquidity };
  });

  const settled = await Promise.allSettled(probes);
  const opportunities = [];

  settled.forEach((result, i) => {
    if (result.status === 'fulfilled') {
      if (result.value) opportunities.push(result.value);
    } else {
      // A pool can exist but not yet be initialized/tradable. That is expected before
      // launch -- but it is recorded rather than silently discarded, so an ABI mismatch
      // or a failing RPC is distinguishable from "not live yet".
      diagnostics.push({
        venue: 'Uniswap V3',
        fee: UNI_FEE_TIERS[i],
        error: result.reason?.shortMessage ?? result.reason?.message ?? String(result.reason)
      });
    }
  });

  return opportunities;
}

export async function inspectAerodrome(ctx) {
  const { aeroFactory, weth, aeroRouterRead, buyWei, minWethLiq, diagnostics, token } = ctx;

  try {
    // A meme-token/WETH pair is a volatile pool, not a stable pool.
    const pool = await aeroFactory.getPool(WETH, token, false);
    if (!pool || pool === ZERO) {
      diagnostics.push({ venue: 'Aerodrome', skipped: 'no pool' });
      return [];
    }

    const wethLiquidity = await weth.balanceOf(pool);
    if (wethLiquidity < minWethLiq) {
      diagnostics.push({
        venue: 'Aerodrome', pool, skipped: 'below MIN_WETH_LIQ', wethLiquidity
      });
      return [];
    }

    const routes = [{ from: WETH, to: token, stable: false, factory: AERO_FACTORY }];
    const amounts = await aeroRouterRead.getAmountsOut(buyWei, routes);
    const quotedOut = amounts.at(-1);
    if (!quotedOut || quotedOut <= 0n) {
      diagnostics.push({ venue: 'Aerodrome', pool, skipped: 'zero quote' });
      return [];
    }

    return [{ venue: 'Aerodrome', pool, stable: false, quotedOut, wethLiquidity, routes }];
  } catch (err) {
    diagnostics.push({
      venue: 'Aerodrome',
      error: err?.shortMessage ?? err?.message ?? String(err)
    });
    return [];
  }
}

/**
 * Quote the round trip: BUY_WEI -> token -> WETH.
 *
 * This does not prove the token is sellable (only a real sell from a holding address
 * does that, and there is nothing to sell before the buy). What it does catch is the
 * two cheap-to-detect failure modes: no reverse route at all, and a sell side priced so
 * far below the buy side that a transfer tax is eating the position.
 */
export async function probeSellPath(ctx, op) {
  const { uniQuoter, aeroRouterRead, diagnostics, token } = ctx;

  try {
    let returned;

    if (op.venue === 'Uniswap V3') {
      const [out] = await uniQuoter.quoteExactInputSingle.staticCall({
        tokenIn: token,
        tokenOut: WETH,
        amountIn: op.quotedOut,
        fee: op.fee,
        sqrtPriceLimitX96: 0n
      });
      returned = out;
    } else {
      const reverse = [{ from: token, to: WETH, stable: false, factory: AERO_FACTORY }];
      const amounts = await aeroRouterRead.getAmountsOut(op.quotedOut, reverse);
      returned = amounts.at(-1);
    }

    if (!returned || returned <= 0n) {
      return { ok: false, reason: 'sell side returns zero', returned: 0n };
    }
    return { ok: true, returned };
  } catch (err) {
    const reason = err?.shortMessage ?? err?.message ?? String(err);
    diagnostics.push({ venue: op.venue, sellProbe: 'failed', error: reason });
    return { ok: false, reason, returned: 0n };
  }
}

/**
 * Quote a much smaller buy on the same venue, to approximate the undisturbed price.
 * Used only to estimate how much of the fill is the trade moving the pool against itself.
 */
export async function probeSpot(ctx, op, probeIn) {
  const { uniQuoter, aeroRouterRead, token } = ctx;
  try {
    if (op.venue === 'Uniswap V3') {
      const [out] = await uniQuoter.quoteExactInputSingle.staticCall({
        tokenIn: WETH, tokenOut: token, amountIn: probeIn, fee: op.fee, sqrtPriceLimitX96: 0n
      });
      return out;
    }
    const amounts = await aeroRouterRead.getAmountsOut(probeIn, op.routes);
    return amounts.at(-1);
  } catch {
    return null;   // no probe means no impact estimate; the caller decides what that means
  }
}

/** Round-trip retention in basis points: 10000 = you get all your ETH back. */
export function retentionBps(buyWei, returnedWei) {
  if (buyWei === 0n) return 0n;
  return (returnedWei * 10_000n) / buyWei;
}
