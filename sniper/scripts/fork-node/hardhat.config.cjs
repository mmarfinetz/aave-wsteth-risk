/**
 * Base mainnet fork for scripts/fork-test.js.
 *
 * Two things are load-bearing and non-obvious:
 *
 * 1. The fork URL points at the local shim, not at Base. Hardhat's forking client does
 *    not honour HTTPS_PROXY, so in a sandboxed environment it cannot reach Base directly.
 * 2. Hardhat ships no hardfork activation history for chain 8453, so any call at or below
 *    the fork block is rejected with "No known hardfork for execution on historical
 *    block". Supplying a history fixes it; so does mining one block, which is what
 *    fork-test.js does on connect.
 */
module.exports = {
  solidity: '0.8.24',
  networks: {
    hardhat: {
      chainId: 8453,
      hardfork: 'cancun',
      forking: {
        url: process.env.SHIM_URL || 'http://127.0.0.1:9545',
        ...(process.env.FORK_BLOCK ? { blockNumber: Number(process.env.FORK_BLOCK) } : {})
      },
      chains: { 8453: { hardforkHistory: { shanghai: 0, cancun: 0 } } }
    }
  }
};
