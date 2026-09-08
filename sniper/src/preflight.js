import { ethers } from 'ethers';
import { LAPTOP, BASE_CHAIN_ID } from './constants.js';

/**
 * Everything that must be true before the watcher is allowed to arm.
 * Any failure here throws, because a sniper that discovers a problem at launch time
 * has already missed the launch.
 */
export async function preflight({ provider, wallet, contracts, config }) {
  const network = await provider.getNetwork();
  if (network.chainId !== BASE_CHAIN_ID) {
    throw new Error(`Wrong chain: expected Base ${BASE_CHAIN_ID}, got ${network.chainId}`);
  }

  const code = await provider.getCode(LAPTOP);
  if (code === '0x') throw new Error('LAPTOP contract has no code on Base');

  const [symbol, decimals, supply, balance] = await Promise.all([
    contracts.laptop.symbol(),
    contracts.laptop.decimals(),
    contracts.laptop.totalSupply(),
    provider.getBalance(wallet.address)
  ]);

  // BUY_ETH must leave room for gas on top; a wallet holding exactly BUY_ETH passes a
  // naive `balance > BUY_ETH` check and then reverts on send.
  const required = config.BUY_WEI + config.GAS_BUFFER_WEI;
  if (balance < required) {
    throw new Error(
      `Wallet holds ${ethers.formatEther(balance)} ETH but needs at least ` +
      `${ethers.formatEther(required)} (BUY_ETH ${ethers.formatEther(config.BUY_WEI)} ` +
      `+ gas buffer ${ethers.formatEther(config.GAS_BUFFER_WEI)})`
    );
  }

  return { symbol, decimals, supply, balance };
}
