import http from 'node:http';
import { WebSocketServer } from 'ws';
import { ethers } from 'ethers';
import {
  LAPTOP, WETH, UNI_V3_FACTORY, UNI_V3_QUOTER, UNI_V3_ROUTER,
  AERO_FACTORY, AERO_ROUTER,
  ERC20_ABI, UNI_FACTORY_ABI, UNI_QUOTER_ABI, UNI_ROUTER_ABI,
  AERO_FACTORY_ABI, AERO_ROUTER_ABI,
  CHAINLINK_ETH_USD, CHAINLINK_ABI
} from '../src/constants.js';

const erc20 = new ethers.Interface(ERC20_ABI);
const uniFactoryIface = new ethers.Interface(UNI_FACTORY_ABI);
const uniQuoterIface = new ethers.Interface(UNI_QUOTER_ABI);
const uniRouterIface = new ethers.Interface(UNI_ROUTER_ABI);
const aeroFactoryIface = new ethers.Interface(AERO_FACTORY_ABI);
const aeroRouterIface = new ethers.Interface(AERO_ROUTER_ABI);
const chainlinkIface = new ethers.Interface(CHAINLINK_ABI);

const lc = (a) => String(a).toLowerCase();
const ZERO_ADDR = ethers.ZeroAddress;

/**
 * Constant-product output with a fee, the x*y=k curve both venues price on.
 *
 * The mock used to return a fixed amount whatever the input, which made it a lookup
 * table rather than a pool: a probe quote at 1/100th the size came back identical, so
 * price impact was invisible. Pricing off reserves means size actually costs something,
 * the way it does on chain.
 */
export function cpQuote({ amountIn, reserveIn, reserveOut, feeBps = 30n }) {
  if (amountIn <= 0n || reserveIn <= 0n || reserveOut <= 0n) return 0n;
  const afterFee = amountIn * (10_000n - feeBps) / 10_000n;
  return (afterFee * reserveOut) / (reserveIn + afterFee);
}

/**
 * A minimal Base node. It answers real JSON-RPC over HTTP so the code under test goes
 * through genuine ethers ABI encoding and decoding rather than hand-fed stubs -- which
 * is the point: a struct field in the wrong order fails here exactly as it would on Base.
 */
export function createMockChain(overrides = {}) {
  const state = {
    chainId: 8453,
    blockNumber: 30_000_000,
    symbol: 'LAPTOP',
    decimals: 18,
    totalSupply: ethers.parseUnits('1000000000', 18),
    walletBalance: ethers.parseEther('1'),
    tokenHasCode: true,
    ethUsdPrice: 440000000000n,       // $4,400.00 at 8 decimals
    ethUsdDecimals: 8,
    ethUsdUpdatedAt: null,            // null = fresh as of the call

    uniPools: {},        // fee -> pool address
    aeroPool: ZERO_ADDR,
    poolWeth: {},        // pool address (lowercase) -> WETH balance
    uniQuote: null,      // ({ fee, amountIn, tokenIn, tokenOut }) -> bigint | null(revert)
    aeroQuote: null,     // ({ amountIn, from, to }) -> bigint | null(revert)
    // pool address -> { weth, token } reserves; drives the constant-product quotes.
    poolReserves: {},
    tokenBalances: {},   // address -> LAPTOP balance
    allowances: {},      // "owner:spender" -> allowance
    uniSwapReverts: false,
    aeroSwapReverts: false,
    receiptStatus: 1,
    ...overrides
  };

  const calls = [];
  const sent = [];

  /** Price off the pool's reserves, in whichever direction is being asked. */
  function reserveQuote({ pool, amountIn, tokenIn, feeBps }) {
    const reserves = state.poolReserves[lc(pool)];
    if (!reserves) return 0n;
    const buyingToken = lc(tokenIn) === lc(WETH);
    return cpQuote({
      amountIn,
      reserveIn: buyingToken ? reserves.weth : reserves.token,
      reserveOut: buyingToken ? reserves.token : reserves.weth,
      feeBps
    });
  }

  const encodeCall = (iface, fn, values) => iface.encodeFunctionResult(fn, values);

  function handleCall(to, data) {
    const target = lc(to);

    if (target === lc(LAPTOP)) {
      const sel = data.slice(0, 10);
      if (sel === erc20.getFunction('symbol').selector) {
        return encodeCall(erc20, 'symbol', [state.symbol]);
      }
      if (sel === erc20.getFunction('decimals').selector) {
        return encodeCall(erc20, 'decimals', [state.decimals]);
      }
      if (sel === erc20.getFunction('totalSupply').selector) {
        return encodeCall(erc20, 'totalSupply', [state.totalSupply]);
      }
      if (sel === erc20.getFunction('balanceOf').selector) {
        const [who] = erc20.decodeFunctionData('balanceOf', data);
        return encodeCall(erc20, 'balanceOf', [state.tokenBalances[lc(who)] ?? 0n]);
      }
      if (sel === erc20.getFunction('allowance').selector) {
        const [owner, spender] = erc20.decodeFunctionData('allowance', data);
        return encodeCall(erc20, 'allowance',
          [state.allowances[`${lc(owner)}:${lc(spender)}`] ?? 0n]);
      }
      if (sel === erc20.getFunction('approve').selector) {
        return encodeCall(erc20, 'approve', [true]);
      }
    }

    if (target === lc(CHAINLINK_ETH_USD)) {
      const sel = data.slice(0, 10);
      if (sel === chainlinkIface.getFunction('decimals').selector) {
        return encodeCall(chainlinkIface, 'decimals', [state.ethUsdDecimals]);
      }
      if (sel === chainlinkIface.getFunction('latestRoundData').selector) {
        const updatedAt = state.ethUsdUpdatedAt ?? Math.floor(Date.now() / 1000);
        return encodeCall(chainlinkIface, 'latestRoundData',
          [1n, state.ethUsdPrice, updatedAt, updatedAt, 1n]);
      }
    }

    if (target === lc(WETH)) {
      const [pool] = erc20.decodeFunctionData('balanceOf', data);
      return encodeCall(erc20, 'balanceOf', [state.poolWeth[lc(pool)] ?? 0n]);
    }

    if (target === lc(UNI_V3_FACTORY)) {
      const [, , fee] = uniFactoryIface.decodeFunctionData('getPool', data);
      return encodeCall(uniFactoryIface, 'getPool',
        [state.uniPools[Number(fee)] ?? ZERO_ADDR]);
    }

    if (target === lc(AERO_FACTORY)) {
      const [, , stable] = aeroFactoryIface.decodeFunctionData('getPool', data);
      return encodeCall(aeroFactoryIface, 'getPool',
        [stable ? ZERO_ADDR : state.aeroPool]);
    }

    if (target === lc(UNI_V3_QUOTER)) {
      const [p] = uniQuoterIface.decodeFunctionData('quoteExactInputSingle', data);
      const fee = Number(p.fee);
      const out = state.uniQuote
        ? state.uniQuote({ tokenIn: p.tokenIn, tokenOut: p.tokenOut, amountIn: p.amountIn, fee })
        : reserveQuote({
            pool: state.uniPools[fee], amountIn: p.amountIn,
            tokenIn: p.tokenIn, feeBps: BigInt(fee) / 100n
          });
      if (out === null) throw new Error('execution reverted: no pool');
      return encodeCall(uniQuoterIface, 'quoteExactInputSingle', [out, 0n, 0, 0n]);
    }

    if (target === lc(AERO_ROUTER)) {
      const sel = data.slice(0, 10);
      if (sel === aeroRouterIface.getFunction('getAmountsOut').selector) {
        const [amountIn, routes] = aeroRouterIface.decodeFunctionData('getAmountsOut', data);
        const out = state.aeroQuote
          ? state.aeroQuote({ amountIn, from: routes[0][0], to: routes[0][1], stable: routes[0][2] })
          : reserveQuote({
              pool: state.aeroPool, amountIn, tokenIn: routes[0][0], feeBps: 30n
            });
        if (out === null) throw new Error('execution reverted');
        return encodeCall(aeroRouterIface, 'getAmountsOut', [[amountIn, out]]);
      }
      if (sel === aeroRouterIface.getFunction('swapExactETHForTokens').selector) {
        if (state.aeroSwapReverts) throw new Error('execution reverted: aerodrome swap');
        const decoded = aeroRouterIface.decodeFunctionData('swapExactETHForTokens', data);
        return encodeCall(aeroRouterIface, 'swapExactETHForTokens', [[0n, decoded[0]]]);
      }
    }

    if (target === lc(UNI_V3_ROUTER)) {
      if (state.uniSwapReverts) throw new Error('execution reverted: uniswap swap');
      const [p] = uniRouterIface.decodeFunctionData('exactInputSingle', data);
      return encodeCall(uniRouterIface, 'exactInputSingle', [p.amountOutMinimum]);
    }

    throw new Error(`mock-chain: unhandled eth_call to ${to} data=${data.slice(0, 10)}`);
  }

  const hex = (n) => '0x' + BigInt(n).toString(16);
  let lastTxHash = '0x' + 'ab'.repeat(32);

  /**
   * Decode a signed, broadcast transaction. Only eth_sendRawTransaction reaches here,
   * so `sent` records what genuinely went to the network -- simulations via eth_call
   * deliberately do not appear, which is what makes "nothing was broadcast" assertions
   * mean something.
   */
  function recordBroadcast(raw) {
    const tx = ethers.Transaction.from(raw);
    lastTxHash = tx.hash;
    const target = lc(tx.to);

    if (target === lc(UNI_V3_ROUTER)) {
      const [p] = uniRouterIface.decodeFunctionData('exactInputSingle', tx.data);
      sent.push({
        venue: 'Uniswap V3', fee: Number(p.fee), amountOutMin: p.amountOutMinimum,
        recipient: p.recipient, amountIn: p.amountIn, value: tx.value
      });
    } else if (target === lc(AERO_ROUTER)) {
      const d = aeroRouterIface.decodeFunctionData('swapExactETHForTokens', tx.data);
      sent.push({
        venue: 'Aerodrome', amountOutMin: d[0], recipient: d[2],
        deadline: d[3], value: tx.value
      });
    } else if (target === lc(LAPTOP)) {
      const [spender, amount] = erc20.decodeFunctionData('approve', tx.data);
      sent.push({ venue: 'approve', spender, amount });
      state.allowances[`${lc(ethers.recoverAddress(
        ethers.Transaction.from(raw).unsignedHash, tx.signature
      ))}:${lc(spender)}`] = amount;
    } else {
      sent.push({ venue: 'unknown', to: tx.to, data: tx.data });
    }
    return lastTxHash;
  }

  function dispatch({ method, params }) {
    calls.push(method);
    switch (method) {
      case 'eth_chainId':      return hex(state.chainId);
      case 'net_version':      return String(state.chainId);
      case 'eth_blockNumber':  return hex(state.blockNumber);
      case 'eth_getCode':      return state.tokenHasCode ? '0x60806040' : '0x';
      case 'eth_getBalance':   return hex(state.walletBalance);
      case 'eth_getTransactionCount': return hex(7);
      case 'eth_gasPrice':     return hex(1_000_000n);
      case 'eth_maxPriorityFeePerGas': return hex(100_000n);
      case 'eth_estimateGas':  return hex(250_000n);
      case 'eth_call':         return handleCall(params[0].to, params[0].data);
      case 'eth_sendRawTransaction': return recordBroadcast(params[0]);
      case 'eth_getBlockByNumber':
        return {
          number: hex(state.blockNumber), hash: '0x' + '11'.repeat(32),
          parentHash: '0x' + '22'.repeat(32), timestamp: hex(Math.floor(Date.now() / 1000)),
          gasLimit: hex(30_000_000), gasUsed: hex(1_000_000),
          baseFeePerGas: hex(1_000_000), miner: ZERO_ADDR, extraData: '0x',
          difficulty: '0x0', totalDifficulty: '0x0', nonce: '0x0000000000000000',
          sha3Uncles: '0x' + '00'.repeat(32), stateRoot: '0x' + '00'.repeat(32),
          receiptsRoot: '0x' + '00'.repeat(32), transactionsRoot: '0x' + '00'.repeat(32),
          logsBloom: '0x' + '00'.repeat(256), size: hex(1000), uncles: [],
          transactions: []
        };
      case 'eth_getTransactionReceipt':
        return {
          transactionHash: lastTxHash, transactionIndex: '0x0',
          blockHash: '0x' + '11'.repeat(32), blockNumber: hex(state.blockNumber),
          from: ZERO_ADDR, to: ZERO_ADDR, cumulativeGasUsed: hex(250_000),
          gasUsed: hex(250_000), contractAddress: null, logs: [],
          logsBloom: '0x' + '00'.repeat(256), status: hex(state.receiptStatus),
          effectiveGasPrice: hex(1_000_000), type: '0x2'
        };
      default:
        throw new Error(`mock-chain: unhandled method ${method}`);
    }
  }

  const server = http.createServer((req, res) => {
    let body = '';
    req.on('data', (c) => { body += c; });
    req.on('end', () => {
      let payload;
      try { payload = JSON.parse(body); } catch { payload = null; }
      const one = (entry) => {
        try {
          return { jsonrpc: '2.0', id: entry.id, result: dispatch(entry) };
        } catch (err) {
          return {
            jsonrpc: '2.0', id: entry.id,
            error: { code: 3, message: err.message, data: '0x' }
          };
        }
      };
      const out = Array.isArray(payload) ? payload.map(one) : one(payload ?? {});
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(out));
    });
  });

  // --- websocket transport -------------------------------------------------------
  // The watcher subscribes to newHeads over a socket, so exercising its reconnect path
  // needs a real ws server that can be told to stop delivering heads.
  let wss = null;
  const sockets = new Set();
  const subscriptions = new Map();   // ws -> subscription id
  let silent = false;

  function blockHeader() {
    return {
      number: hex(state.blockNumber), hash: '0x' + '11'.repeat(32),
      parentHash: '0x' + '22'.repeat(32),
      timestamp: hex(Math.floor(Date.now() / 1000)),
      gasLimit: hex(30_000_000), gasUsed: hex(1_000_000),
      baseFeePerGas: hex(1_000_000), miner: ZERO_ADDR, extraData: '0x',
      difficulty: '0x0', nonce: '0x0000000000000000',
      sha3Uncles: '0x' + '00'.repeat(32), stateRoot: '0x' + '00'.repeat(32),
      receiptsRoot: '0x' + '00'.repeat(32), transactionsRoot: '0x' + '00'.repeat(32),
      logsBloom: '0x' + '00'.repeat(256)
    };
  }

  return {
    state,
    calls,
    sent,
    get socketCount() { return sockets.size; },

    /** Produce a block and push it to every live subscriber. */
    mineBlock() {
      state.blockNumber += 1;
      if (silent) return state.blockNumber;
      for (const ws of sockets) {
        const id = subscriptions.get(ws);
        if (!id || ws.readyState !== ws.OPEN) continue;
        ws.send(JSON.stringify({
          jsonrpc: '2.0', method: 'eth_subscription',
          params: { subscription: id, result: blockHeader() }
        }));
      }
      return state.blockNumber;
    },

    /**
     * Stop delivering heads while leaving the socket open. This is the failure the
     * watchdog exists for -- the process stays connected and simply goes deaf.
     */
    goSilent() { silent = true; },
    resume() { silent = false; },

    async listenWs() {
      const server = http.createServer();
      await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
      wss = new WebSocketServer({ server });
      wss.on('connection', (ws) => {
        sockets.add(ws);
        ws.on('close', () => { sockets.delete(ws); subscriptions.delete(ws); });
        ws.on('message', (raw) => {
          let payload;
          try { payload = JSON.parse(raw.toString()); } catch { return; }
          const answer = (entry) => {
            if (entry.method === 'eth_subscribe') {
              calls.push('eth_subscribe');
              const id = '0x' + (subscriptions.size + 1).toString(16).padStart(2, '0');
              subscriptions.set(ws, id);
              return { jsonrpc: '2.0', id: entry.id, result: id };
            }
            if (entry.method === 'eth_unsubscribe') {
              subscriptions.delete(ws);
              return { jsonrpc: '2.0', id: entry.id, result: true };
            }
            try {
              return { jsonrpc: '2.0', id: entry.id, result: dispatch(entry) };
            } catch (err) {
              return {
                jsonrpc: '2.0', id: entry.id,
                error: { code: 3, message: err.message, data: '0x' }
              };
            }
          };
          const out = Array.isArray(payload) ? payload.map(answer) : answer(payload);
          ws.send(JSON.stringify(out));
        });
      });
      const { port } = server.address();
      this._wsHttp = server;
      return `ws://127.0.0.1:${port}`;
    },

    async closeWs() {
      for (const ws of sockets) { try { ws.terminate(); } catch { /* gone */ } }
      sockets.clear();
      if (wss) await new Promise((r) => wss.close(r));
      if (this._wsHttp) await new Promise((r) => this._wsHttp.close(r));
    },

    async listen() {
      await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
      const { port } = server.address();
      // Static network + no polling keeps tests deterministic and fast.
      const provider = new ethers.JsonRpcProvider(
        `http://127.0.0.1:${port}`,
        { chainId: state.chainId, name: 'base' },
        { staticNetwork: true, pollingInterval: 10 }
      );
      return provider;
    },
    async close() {
      await new Promise((resolve) => server.close(resolve));
    }
  };
}
