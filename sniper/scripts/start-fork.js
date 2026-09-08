#!/usr/bin/env node
/**
 * Bring up a Base fork for scripts/fork-test.js, in one command.
 *
 *   BASE_RPC=https://... npm run fork-node
 *   # then, in another shell:
 *   FORK_RPC=http://127.0.0.1:8545 TOKEN=0x... npm run fork-test
 *
 * foundry/anvil would be the obvious tool, but foundryup downloads its binaries from
 * GitHub release assets, which this sandbox's GitHub proxy serves only for repositories
 * attached to the session -- so it 403s. Hardhat installs from npm and forks equally
 * well for this purpose.
 */
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const UPSTREAM = process.env.BASE_RPC ?? 'https://mainnet.base.org';
const SHIM_PORT = process.env.SHIM_PORT ?? '9545';
const NODE_PORT = process.env.NODE_PORT ?? '8545';

const children = [];
const stop = () => { for (const c of children) { try { c.kill(); } catch { /* gone */ } } };
process.on('SIGINT', () => { stop(); process.exit(0); });
process.on('SIGTERM', () => { stop(); process.exit(0); });

console.log(`shim  : 127.0.0.1:${SHIM_PORT} -> ${UPSTREAM.replace(/\/v2\/.*$/, '/v2/<key>')}`);
const shim = spawn(process.execPath, [path.join(here, 'fork-node', 'rpc-shim.cjs')], {
  env: { ...process.env, UPSTREAM_RPC: UPSTREAM, SHIM_PORT },
  stdio: 'inherit'
});
children.push(shim);

// Give the shim a moment to bind before Hardhat starts pulling state through it.
await new Promise((r) => setTimeout(r, 1500));

console.log(`fork  : 127.0.0.1:${NODE_PORT}`);
const node = spawn('npx', ['hardhat', 'node', '--hostname', '127.0.0.1', '--port', NODE_PORT], {
  cwd: here,
  env: {
    ...process.env,
    HARDHAT_CONFIG: path.join(here, 'fork-node', 'hardhat.config.cjs'),
    SHIM_URL: `http://127.0.0.1:${SHIM_PORT}`
  },
  stdio: 'inherit'
});
children.push(node);
node.on('exit', (code) => { stop(); process.exit(code ?? 0); });
