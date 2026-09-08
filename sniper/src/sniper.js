import { ethers } from 'ethers';
import { WETH, LAPTOP } from './constants.js';
import { maxBigInt, slippageMin } from './config.js';
import { inspectUniswapV3, inspectAerodrome, probeSellPath, retentionBps } from './venues.js';

// If the round trip returns less than this, treat the token as untradable rather than
// merely expensive. 5000 bps = you would get half your ETH back selling straight away.
const MIN_RETENTION_BPS = 5_000n;

export class Sniper {
  constructor({ contracts, wallet, config, logger = console, now = () => Date.now() }) {
    this.contracts = contracts;
    this.wallet = wallet;
    this.config = config;
    this.logger = logger;
    this.now = now;

    this.tokenDecimals = 18;
    this.absoluteMinOut = 0n;
    this.busy = false;
    this.bought = false;
    this.lastFingerprint = '';
  }

  setTokenDecimals(decimals) {
    this.tokenDecimals = Number(decimals);
    if (this.config.MIN_TOKENS_OUT) {
      this.absoluteMinOut = ethers.parseUnits(
        String(this.config.MIN_TOKENS_OUT).trim(),
        this.tokenDecimals
      );
    }
  }

  probeContext(diagnostics) {
    return {
      ...this.contracts,
      buyWei: this.config.BUY_WEI,
      minWethLiq: this.config.MIN_WETH_LIQ,
      diagnostics
    };
  }

  async scan(blockNumber) {
    const { config, logger } = this;

    if (this.now() < config.EARLIEST_BUY_MS) {
      return { status: 'before-earliest-buy' };
    }

    const diagnostics = [];
    const ctx = this.probeContext(diagnostics);
    const quotedAt = this.now();

    // Both venues are probed concurrently; inside Uniswap all four fee tiers are too.
    const [uni, aero] = await Promise.all([
      inspectUniswapV3(ctx),
      inspectAerodrome(ctx)
    ]);
    const opportunities = [...uni, ...aero];

    if (opportunities.length === 0) {
      return { status: 'no-liquidity', diagnostics };
    }

    // Among pools clearing the liquidity threshold, prefer the best current quote.
    opportunities.sort((a, b) =>
      a.quotedOut > b.quotedOut ? -1 : a.quotedOut < b.quotedOut ? 1 : 0
    );
    const op = opportunities[0];

    // Dedupe on venue+pool only. Including the quote made the fingerprint change every
    // block once a pool was live, which reprinted the banner forever.
    const fingerprint = `${op.venue}:${op.pool}`;
    if (fingerprint !== this.lastFingerprint) {
      this.lastFingerprint = fingerprint;
      logger.log(`\n[block ${blockNumber}] Tradable liquidity found`);
      logger.log(`venue:       ${op.venue}`);
      logger.log(`pool:        ${op.pool}`);
      logger.log(`pool WETH:   ${ethers.formatEther(op.wethLiquidity)}`);
      logger.log(`buy ETH:     ${ethers.formatEther(config.BUY_WEI)}`);
      logger.log(`quoted out:  ${ethers.formatUnits(op.quotedOut, this.tokenDecimals)} LAPTOP`);
    }

    if (!config.LIVE) {
      logger.log('LIVE=false - observation only; no transaction sent.');
      return { status: 'observed', op, diagnostics };
    }

    // Belt and braces: config.js already refuses LIVE without MIN_TOKENS_OUT at startup.
    if (this.absoluteMinOut === 0n) {
      return { status: 'blocked', reason: 'missing-absolute-min-out', op };
    }

    if (config.REQUIRE_SELL_PATH) {
      const sell = await probeSellPath(ctx, op);
      if (!sell.ok) {
        logger.error(`Sell-path probe failed (${sell.reason}) - not buying.`);
        return { status: 'blocked', reason: 'no-sell-path', detail: sell.reason, op };
      }
      const retention = retentionBps(config.BUY_WEI, sell.returned);
      if (retention < MIN_RETENTION_BPS) {
        logger.error(
          `Round trip returns ${Number(retention) / 100}% of input - below the ` +
          `${Number(MIN_RETENTION_BPS) / 100}% floor. Not buying.`
        );
        return { status: 'blocked', reason: 'low-retention', retention, op };
      }
      op.retentionBps = retention;
    }

    // Refuse to size against a quote that has gone stale while probing.
    const age = this.now() - quotedAt;
    if (age > config.MAX_SCAN_STALENESS_MS) {
      logger.error(`Quote is ${age}ms old (limit ${config.MAX_SCAN_STALENESS_MS}ms) - re-quoting next block.`);
      return { status: 'stale', age, op };
    }

    logger.log(`Attempting ${op.venue} purchase...`);
    const tx = op.venue === 'Uniswap V3'
      ? await this.sendUniswap(op)
      : await this.sendAerodrome(op);

    logger.log(`submitted: ${tx.hash}`);
    const receipt = await tx.wait();
    if (receipt?.status !== 1) throw new Error('Transaction reverted');

    this.bought = true;
    logger.log(`SUCCESS in block ${receipt.blockNumber}: ${tx.hash}`);
    return { status: 'bought', op, tx, receipt };
  }

  minOutFor(op) {
    return maxBigInt(
      slippageMin(op.quotedOut, this.config.SLIPPAGE_BPS),
      this.absoluteMinOut
    );
  }

  async sendUniswap(op) {
    const { uniRouter } = this.contracts;
    const params = {
      tokenIn: WETH,
      tokenOut: LAPTOP,
      fee: op.fee,
      recipient: this.wallet.address,
      amountIn: this.config.BUY_WEI,
      amountOutMinimum: this.minOutFor(op),
      sqrtPriceLimitX96: 0n
    };

    // Full transaction simulation from the actual buying wallet before anything is sent.
    await uniRouter.exactInputSingle.staticCall(params, { value: this.config.BUY_WEI });
    const gas = await uniRouter.exactInputSingle.estimateGas(params, { value: this.config.BUY_WEI });
    return uniRouter.exactInputSingle(params, {
      value: this.config.BUY_WEI,
      gasLimit: (gas * 120n) / 100n
    });
  }

  async sendAerodrome(op) {
    const { aeroRouter } = this.contracts;
    const amountOutMinimum = this.minOutFor(op);
    const deadline = BigInt(Math.floor(this.now() / 1000) + 60);
    const args = [amountOutMinimum, op.routes, this.wallet.address, deadline];
    const overrides = { value: this.config.BUY_WEI };

    await aeroRouter.swapExactETHForTokens.staticCall(...args, overrides);
    const gas = await aeroRouter.swapExactETHForTokens.estimateGas(...args, overrides);
    return aeroRouter.swapExactETHForTokens(...args, {
      ...overrides,
      gasLimit: (gas * 120n) / 100n
    });
  }
}

export { MIN_RETENTION_BPS };
