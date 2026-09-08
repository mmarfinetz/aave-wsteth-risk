import http from 'node:http';
import { ethers } from 'ethers';
import { ORDER_TYPES, cowDomain, computeOrderUid } from '../src/cow.js';

/**
 * A stand-in CoW orderbook.
 *
 * It verifies the EIP-712 signature by recovering the signer from the Order struct, so a
 * field in the wrong position or a bad domain fails here the same way solvers would
 * reject it -- rather than being waved through by a stub that only checks shape.
 */
export function createMockCow({ chainId = 8453, network = 'base', noRoute = false } = {}) {
  const state = {
    noRoute,
    quoteBuyAmount: ethers.parseEther('0.02'),
    quoteFeeAmount: ethers.parseEther('0.0001'),
    appData: ethers.ZeroHash,
    appDataHash: null
  };

  const orders = new Map();   // uid -> { order, from, status, executed... }
  const rejections = [];
  const requests = [];

  function handleQuote(body) {
    if (state.noRoute) {
      return {
        status: 404,
        payload: { errorType: 'NoLiquidity', description: 'no route found for sell token' }
      };
    }
    const sellAmountBeforeFee = BigInt(body.sellAmountBeforeFee);
    const fee = state.quoteFeeAmount;
    return {
      status: 200,
      payload: {
        quote: {
          sellToken: body.sellToken,
          buyToken: body.buyToken,
          receiver: body.receiver ?? body.from,
          sellAmount: (sellAmountBeforeFee - fee).toString(),
          buyAmount: state.quoteBuyAmount.toString(),
          validTo: Math.floor(Date.now() / 1000) + 600,
          appData: state.appData,
          feeAmount: fee.toString(),
          kind: 'sell',
          partiallyFillable: false,
          sellTokenBalance: 'erc20',
          buyTokenBalance: 'erc20'
        },
        from: body.from,
        expiration: new Date(Date.now() + 60_000).toISOString(),
        id: 12345,
        ...(state.appDataHash ? { appDataHash: state.appDataHash } : {})
      }
    };
  }

  function handlePlaceOrder(body) {
    const { signature, signingScheme, from, appDataHash, ...rest } = body;

    // Reconstruct exactly the struct that should have been signed.
    const order = {
      sellToken: rest.sellToken,
      buyToken: rest.buyToken,
      receiver: rest.receiver,
      sellAmount: rest.sellAmount,
      buyAmount: rest.buyAmount,
      validTo: rest.validTo,
      appData: appDataHash ?? rest.appData,
      feeAmount: rest.feeAmount,
      kind: rest.kind,
      partiallyFillable: rest.partiallyFillable,
      sellTokenBalance: rest.sellTokenBalance,
      buyTokenBalance: rest.buyTokenBalance
    };

    let recovered;
    try {
      recovered = ethers.verifyTypedData(cowDomain(chainId), ORDER_TYPES, order, signature);
    } catch (err) {
      rejections.push({ reason: 'signature-undecodable', error: err.message });
      return { status: 400, payload: { errorType: 'InvalidSignature', description: err.message } };
    }

    if (recovered.toLowerCase() !== String(from).toLowerCase()) {
      rejections.push({ reason: 'signer-mismatch', recovered, from });
      return {
        status: 400,
        payload: { errorType: 'InvalidSignature', description: `recovered ${recovered}, expected ${from}` }
      };
    }
    if (signingScheme !== 'eip712') {
      return { status: 400, payload: { errorType: 'UnsupportedSigningScheme' } };
    }

    const uid = computeOrderUid(order, from, chainId);
    orders.set(uid.toLowerCase(), {
      order, from, status: 'open',
      executedSellAmount: '0', executedBuyAmount: '0'
    });
    return { status: 201, payload: uid };
  }

  const server = http.createServer((req, res) => {
    let body = '';
    req.on('data', (c) => { body += c; });
    req.on('end', () => {
      const url = new URL(req.url, 'http://localhost');
      const path = url.pathname.replace(`/${network}/api/v1`, '');
      requests.push(`${req.method} ${path}`);

      let parsed = null;
      try { parsed = body ? JSON.parse(body) : null; } catch { /* leave null */ }

      let result;
      if (req.method === 'POST' && path === '/quote') {
        result = handleQuote(parsed);
      } else if (req.method === 'POST' && path === '/orders') {
        result = handlePlaceOrder(parsed);
      } else if (req.method === 'GET' && path.startsWith('/orders/')) {
        const uid = path.slice('/orders/'.length).toLowerCase();
        const found = orders.get(uid);
        result = found
          ? { status: 200, payload: { uid, ...found.order, status: found.status,
              executedSellAmount: found.executedSellAmount,
              executedBuyAmount: found.executedBuyAmount } }
          : { status: 404, payload: { errorType: 'NotFound' } };
      } else if (req.method === 'DELETE' && path === '/orders') {
        for (const uid of parsed?.orderUids ?? []) {
          const found = orders.get(String(uid).toLowerCase());
          if (found) found.status = 'cancelled';
        }
        result = { status: 200, payload: 'Cancelled' };
      } else {
        result = { status: 404, payload: { errorType: 'NotFound', description: path } };
      }

      res.writeHead(result.status, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(result.payload));
    });
  });

  return {
    state,
    orders,
    rejections,
    requests,
    /** Simulate a solver settling a rung, fully or partially. */
    fill(uid, { sellAmount, buyAmount, partial = false } = {}) {
      const found = orders.get(String(uid).toLowerCase());
      if (!found) throw new Error(`no such order ${uid}`);
      found.status = partial ? 'open' : 'fulfilled';
      found.executedSellAmount = String(sellAmount ?? found.order.sellAmount);
      found.executedBuyAmount = String(buyAmount ?? found.order.buyAmount);
      return found;
    },
    async listen() {
      await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
      return `http://127.0.0.1:${server.address().port}`;
    },
    async close() {
      await new Promise((resolve) => server.close(resolve));
    }
  };
}
