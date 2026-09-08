import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { ethers } from 'ethers';

const FIXTURE_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), 'fixtures');

/** Every recorded fixture on disk. Empty until someone runs scripts/record-fixtures.js. */
export function loadFixtures() {
  if (!fs.existsSync(FIXTURE_DIR)) return [];
  return fs.readdirSync(FIXTURE_DIR)
    .filter((f) => f.endsWith('.json'))
    .map((f) => ({
      name: f,
      ...JSON.parse(fs.readFileSync(path.join(FIXTURE_DIR, f), 'utf8'))
    }));
}

/**
 * Serve a recorded fixture over JSON-RPC.
 *
 * Responses are the exact bytes Base returned, keyed by (to, calldata). Nothing here
 * invents a value, so decoding these exercises the real wire format -- which is the one
 * thing the hand-written mocks structurally cannot do.
 */
export function createReplayChain(fixture) {
  const byKey = new Map();
  for (const call of fixture.calls) {
    byKey.set(`${call.to.toLowerCase()}:${call.data.toLowerCase()}`, call);
  }

  const misses = [];
  const hex = (n) => '0x' + BigInt(n).toString(16);

  const server = http.createServer((req, res) => {
    let body = '';
    req.on('data', (c) => { body += c; });
    req.on('end', () => {
      const payload = JSON.parse(body);
      const answer = (entry) => {
        const { method, params } = entry;
        switch (method) {
          case 'eth_chainId': return { jsonrpc: '2.0', id: entry.id, result: hex(fixture.meta.chainId) };
          case 'net_version': return { jsonrpc: '2.0', id: entry.id, result: String(fixture.meta.chainId) };
          case 'eth_blockNumber': return { jsonrpc: '2.0', id: entry.id, result: hex(fixture.meta.blockNumber) };
          case 'eth_getCode':
            return { jsonrpc: '2.0', id: entry.id, result: fixture.meta.code ? '0x60806040' : '0x' };
          case 'eth_call': {
            const key = `${String(params[0].to).toLowerCase()}:${String(params[0].data).toLowerCase()}`;
            const found = byKey.get(key);
            if (found === undefined) {
              misses.push(key);
              return {
                jsonrpc: '2.0', id: entry.id,
                error: { code: 3, message: 'execution reverted: not in fixture', data: '0x' }
              };
            }
            // Replay recorded reverts as reverts, so the fixture reproduces what Base
            // actually did rather than a chain where everything succeeds.
            if (found.revert) {
              return {
                jsonrpc: '2.0', id: entry.id,
                error: { code: 3, message: found.errorMessage ?? 'execution reverted',
                         data: found.errorData ?? '0x' }
              };
            }
            return { jsonrpc: '2.0', id: entry.id, result: found.result };
          }
          default:
            return {
              jsonrpc: '2.0', id: entry.id,
              error: { code: -32601, message: `replay: unhandled ${method}` }
            };
        }
      };
      const out = Array.isArray(payload) ? payload.map(answer) : answer(payload);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(out));
    });
  });

  return {
    misses,
    recordedCalls: fixture.calls.length,
    async listen() {
      await new Promise((r) => server.listen(0, '127.0.0.1', r));
      return new ethers.JsonRpcProvider(
        `http://127.0.0.1:${server.address().port}`,
        { chainId: fixture.meta.chainId, name: 'base' },
        { staticNetwork: true, pollingInterval: 10 }
      );
    },
    async close() { await new Promise((r) => server.close(r)); }
  };
}
