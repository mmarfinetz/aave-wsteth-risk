import { ethers } from 'ethers';
import { buildContracts } from './wiring.js';
import { preflight as defaultPreflight } from './preflight.js';
import { Sniper } from './sniper.js';

/**
 * Owns the socket lifecycle.
 *
 * A websocket that dies quietly is the worst failure mode for a launch watcher: the
 * process stays up, looks healthy, and simply never fires again. Base produces a block
 * roughly every 2 seconds, so silence past the watchdog window means the socket is gone
 * rather than the chain being idle -- and the only safe response is to rebuild it.
 */
export class Watcher {
  constructor({
    config,
    logger = console,
    makeProvider = (url) => new ethers.WebSocketProvider(url),
    preflightFn = defaultPreflight,
    now = () => Date.now(),
    onBought
  }) {
    this.config = config;
    this.logger = logger;
    this.makeProvider = makeProvider;
    this.preflightFn = preflightFn;
    this.now = now;
    this.onBought = onBought;

    this.provider = null;
    this.sniper = null;
    this.watchdog = null;
    this.lastBlockAt = 0;
    this.reconnects = 0;
    this.scans = 0;
    this.stopped = false;
    this.info = null;
  }

  async start() {
    this.stopped = false;
    await this.connect();
    return this.info;
  }

  async connect() {
    this.provider = this.makeProvider(this.config.BASE_WSS);
    const wallet = new ethers.Wallet(this.config.PRIVATE_KEY, this.provider);
    const contracts = buildContracts(this.provider, wallet);

    this.info = await this.preflightFn({
      provider: this.provider, wallet, contracts, config: this.config
    });

    // `bought` must survive a reconnect, or a dropped socket becomes a second buy.
    const alreadyBought = this.sniper?.bought ?? false;
    this.sniper = new Sniper({
      contracts, wallet, config: this.config, logger: this.logger, now: this.now
    });
    this.sniper.setTokenDecimals(this.info.decimals);
    this.sniper.bought = alreadyBought;

    this.lastBlockAt = this.now();
    this.provider.on('block', (blockNumber) => this.handleBlock(blockNumber));
    this.startWatchdog();
    return this.info;
  }

  async handleBlock(blockNumber) {
    this.lastBlockAt = this.now();
    if (this.stopped || this.sniper.busy || this.sniper.bought) return;

    this.sniper.busy = true;
    try {
      this.scans += 1;
      const result = await this.sniper.scan(blockNumber);
      if (result.status === 'bought') {
        await this.stop();
        this.onBought?.(result);
      }
    } catch (err) {
      this.logger.error(`[block ${blockNumber}] ${err.shortMessage ?? err.message ?? err}`);
    } finally {
      this.sniper.busy = false;
    }
  }

  startWatchdog() {
    this.clearWatchdog();
    const period = Math.max(1_000, Math.floor(this.config.BLOCK_WATCHDOG_MS / 4));
    this.watchdog = setInterval(() => this.checkLiveness(), period);
    this.watchdog.unref?.();
  }

  clearWatchdog() {
    if (this.watchdog) clearInterval(this.watchdog);
    this.watchdog = null;
  }

  async checkLiveness() {
    if (this.stopped) return;
    const silence = this.now() - this.lastBlockAt;
    if (silence < this.config.BLOCK_WATCHDOG_MS) return;

    this.reconnects += 1;
    this.logger.error(`No block for ${silence}ms - reconnecting (attempt ${this.reconnects})`);
    this.clearWatchdog();
    try { await this.provider?.destroy(); } catch { /* already gone */ }

    try {
      await this.connect();
      this.logger.log(`Reconnected (#${this.reconnects})`);
    } catch (err) {
      this.logger.error(`Reconnect failed: ${err.shortMessage ?? err.message ?? err}`);
      if (this.stopped) return;
      // Keep retrying on a bounded backoff: missing the launch is worse than churn.
      const delay = Math.min(30_000, 2_000 * this.reconnects);
      const timer = setTimeout(() => {
        if (!this.stopped) this.checkLiveness();
      }, delay);
      timer.unref?.();
      this.startWatchdog();
    }
  }

  async stop() {
    this.stopped = true;
    this.clearWatchdog();
    try { await this.provider?.destroy(); } catch { /* already gone */ }
  }
}
