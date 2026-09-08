import 'dotenv/config';
import fs from 'node:fs';
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
  onBought: (result) => {
    // The exit ladder prices off this, so record what actually filled.
    const position = {
      token: LAPTOP,
      venue: result.op.venue,
      pool: result.op.pool,
      entryWei: config.BUY_WEI.toString(),
      tokensHeld: result.tokensHeld?.toString() ?? '0',
      txHash: result.txHash,
      blockNumber: result.blockNumber,
      filledAt: new Date().toISOString()
    };
    try {
      fs.writeFileSync(config.POSITION_FILE, JSON.stringify(position, null, 2));
      console.log(`\nPosition written to ${config.POSITION_FILE}`);
      console.log('Next: node scripts/exit.js --plan');
    } catch (err) {
      console.error(`Could not write ${config.POSITION_FILE}: ${err.message}`);
      console.error(`Record this manually: ${JSON.stringify(position)}`);
    }
    process.exit(0);
  }
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
