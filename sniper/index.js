import 'dotenv/config';
import { ethers } from 'ethers';
import { loadConfig } from './src/config.js';
import { Watcher } from './src/watcher.js';
import { LAPTOP } from './src/constants.js';

const config = loadConfig();

function banner(info) {
  console.log('Base launch watcher armed');
  console.log(`token:          ${LAPTOP}`);
  console.log(`symbol:         ${info.symbol}`);
  console.log(`total supply:   ${ethers.formatUnits(info.supply, info.decimals)}`);
  console.log(`wallet ETH:     ${ethers.formatEther(info.balance)}`);
  console.log(`LIVE:           ${config.LIVE}`);
  console.log(`BUY_ETH:        ${ethers.formatEther(config.BUY_WEI)}`);
  console.log(`MIN_WETH_LIQ:   ${ethers.formatEther(config.MIN_WETH_LIQ)}`);
  console.log(`SLIPPAGE:       ${Number(config.SLIPPAGE_BPS) / 100}%`);
  console.log(`EARLIEST_BUY:   ${new Date(config.EARLIEST_BUY_MS).toISOString()}`);
  console.log(`MIN_TOKENS_OUT: ${config.MIN_TOKENS_OUT ?? '(unset)'}`);
  console.log(`sell-path gate: ${config.REQUIRE_SELL_PATH}`);
  if (!config.LIVE) console.log('\nObservation mode: no transaction will be sent.');
}

const watcher = new Watcher({
  config,
  onBought: () => process.exit(0)
});

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, async () => {
    console.log(`\n${signal} - shutting down.`);
    await watcher.stop();
    process.exit(0);
  });
}

try {
  const info = await watcher.start();
  banner(info);
} catch (err) {
  console.error(err.message ?? err);
  await watcher.stop();
  process.exit(1);
}
