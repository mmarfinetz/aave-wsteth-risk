/**
 * Localhost JSON-RPC shim.
 *
 * Hardhat's forking client does not honour HTTPS_PROXY, and the sandbox refuses direct
 * egress, so the fork URL points here instead: plain HTTP on localhost (exempt from the
 * proxy), forwarded upstream through the proxy that does work. Nothing about the payload
 * is inspected or changed.
 */
const http = require('node:http');
const { fetch: undiciFetch, ProxyAgent } = require('undici');

const UPSTREAM = process.env.UPSTREAM_RPC || 'https://mainnet.base.org';
const PROXY = process.env.HTTPS_PROXY || process.env.https_proxy;
const PORT = Number(process.env.SHIM_PORT || 9545);
const dispatcher = PROXY ? new ProxyAgent(PROXY) : undefined;

let forwarded = 0;
let failed = 0;

const server = http.createServer((req, res) => {
  let body = '';
  req.on('data', (c) => { body += c; });
  req.on('end', async () => {
    try {
      const upstream = await undiciFetch(UPSTREAM, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body,
        ...(dispatcher ? { dispatcher } : {})
      });
      const text = await upstream.text();
      forwarded += 1;
      if (forwarded % 250 === 0) console.log(`[shim] ${forwarded} forwarded, ${failed} failed`);
      res.writeHead(upstream.status, { 'content-type': 'application/json' });
      res.end(text);
    } catch (err) {
      failed += 1;
      res.writeHead(502, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ jsonrpc: '2.0', id: null,
        error: { code: -32603, message: `shim upstream failed: ${err.message}` } }));
    }
  });
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`[shim] 127.0.0.1:${PORT} -> ${UPSTREAM}${PROXY ? ' (via proxy)' : ''}`);
});
