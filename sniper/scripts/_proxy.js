/**
 * Route ethers through an HTTP proxy when one is configured.
 *
 * Node's http/https stack ignores HTTPS_PROXY (curl does not), so in a sandboxed
 * environment ethers bypasses the gateway and the request is refused. This is a
 * scripts-only concern: nothing under src/ depends on it, and with no proxy set every
 * function here is a no-op.
 */
import { ethers } from 'ethers';

const PROXY = process.env.HTTPS_PROXY ?? process.env.https_proxy;

let agent = null;
if (PROXY) {
  const { HttpsProxyAgent } = await import('https-proxy-agent');
  agent = new HttpsProxyAgent(PROXY);
}

/** A FetchRequest for `url` that honours the proxy when there is one. */
export function proxiedRequest(url) {
  const request = new ethers.FetchRequest(url);
  if (agent) request.getUrlFunc = ethers.FetchRequest.createGetUrlFunc({ agent });
  return request;
}

export function proxiedProvider(url, network = { chainId: 8453, name: 'base' }) {
  return new ethers.JsonRpcProvider(proxiedRequest(url), network, { staticNetwork: true });
}

export const proxyInUse = Boolean(agent);
