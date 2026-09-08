import { ethers } from 'ethers';
import {
  CowClient, LIMIT_ORDER_FEE, computeOrderUid, signOrder, ensureAllowance, ETH_SENTINEL
} from './cow.js';

const BPS = 10_000n;

/**
 * Parse a ladder spec like "3x:25,5x:25,10x:50".
 *
 * Each rung is <multiple>x:<percent of the position>. Percentages are of the ORIGINAL
 * position, not of what is left, so they read the way people actually think about a
 * take-profit plan.
 */
export function parseLadder(spec) {
  const text = String(spec ?? '').trim();
  if (!text) throw new Error('Ladder spec is empty');

  const rungs = text.split(',').map((raw, i) => {
    const part = raw.trim();
    const match = /^(\d+(?:\.\d+)?)x:(\d+(?:\.\d+)?)$/i.exec(part);
    if (!match) {
      throw new Error(`Rung ${i + 1} ("${part}") is not <multiple>x:<percent>, e.g. 3x:25`);
    }
    const multipleBps = BigInt(Math.round(Number(match[1]) * 10_000));
    const pctBps = BigInt(Math.round(Number(match[2]) * 100));
    if (multipleBps <= 0n) throw new Error(`Rung ${i + 1}: multiple must be > 0`);
    if (pctBps <= 0n) throw new Error(`Rung ${i + 1}: percent must be > 0`);
    return { multipleBps, pctBps, label: part };
  });

  const total = rungs.reduce((sum, r) => sum + r.pctBps, 0n);
  if (total > BPS) {
    throw new Error(
      `Ladder sells ${Number(total) / 100}% of the position; it cannot exceed 100%`
    );
  }
  return rungs;
}

/**
 * Turn a ladder into concrete sell/buy amounts.
 *
 * buyAmount is derived from the entry price: a 3x rung asks for three times the ETH per
 * token that was actually paid. All arithmetic stays in bigint -- a rounding slip here
 * is a rung that rests at the wrong price and either never fills or fills too cheap.
 */
export function buildRungs({ totalTokens, entryWei, ladder }) {
  if (totalTokens <= 0n) throw new Error('totalTokens must be positive');
  if (entryWei <= 0n) throw new Error('entryWei must be positive');

  const rungs = [];
  let allocated = 0n;
  const fullSweep = ladder.reduce((s, r) => s + r.pctBps, 0n) === BPS;

  ladder.forEach((rung, i) => {
    const isLast = i === ladder.length - 1;
    // Give the final rung the remainder so integer division cannot strand dust.
    const sellAmount = (isLast && fullSweep)
      ? totalTokens - allocated
      : (totalTokens * rung.pctBps) / BPS;
    allocated += sellAmount;

    const buyAmount = (sellAmount * entryWei * rung.multipleBps) / (totalTokens * BPS);
    if (sellAmount <= 0n) throw new Error(`Rung ${rung.label} rounds to zero tokens`);
    if (buyAmount <= 0n) throw new Error(`Rung ${rung.label} rounds to zero proceeds`);

    rungs.push({ ...rung, sellAmount, buyAmount });
  });

  return rungs;
}

/**
 * The appData the API expects.
 *
 * The signed struct always takes a bytes32, but the orderbook has carried appData as
 * both a bare hash and a document plus a separate hash, so both shapes are handled and
 * whatever the quote returned is echoed back verbatim.
 */
export function resolveAppData(quoteResponse) {
  const quote = quoteResponse?.quote ?? {};
  const hash = quoteResponse?.appDataHash ?? quote.appDataHash;
  const doc = quoteResponse?.appData ?? quote.appData;

  if (hash && ethers.isHexString(hash, 32)) return { signed: hash, posted: doc ?? hash };
  if (doc && ethers.isHexString(doc, 32)) return { signed: doc, posted: doc };
  return { signed: ethers.ZeroHash, posted: ethers.ZeroHash };
}

export class ExitManager {
  constructor({
    client, wallet, token, chainId = 8453,
    sellToken, buyToken, receiver,
    ttlSeconds = 3 * 24 * 60 * 60,
    partiallyFillable = true,
    logger = console,
    now = () => Date.now()
  }) {
    this.client = client ?? new CowClient({ chainId });
    this.wallet = wallet;
    this.token = token;
    this.chainId = chainId;
    this.sellToken = sellToken;
    this.buyToken = buyToken;
    this.receiver = receiver ?? wallet.address;
    this.ttlSeconds = ttlSeconds;
    this.partiallyFillable = partiallyFillable;
    this.logger = logger;
    this.now = now;
    this.placed = [];
  }

  /**
   * Ask CoW for a quote on the whole position. Doubles as the readiness check: if no
   * solver can route the token, this is where that shows up -- before any approval.
   */
  async readiness(totalTokens) {
    try {
      const response = await this.client.quoteSell({
        sellToken: this.sellToken,
        buyToken: this.buyToken,
        sellAmountBeforeFee: totalTokens,
        from: this.wallet.address,
        receiver: this.receiver
      });
      return {
        routable: true,
        quote: response,
        buyAmount: BigInt(response.quote.buyAmount),
        feeAmount: BigInt(response.quote.feeAmount ?? 0)
      };
    } catch (err) {
      return { routable: false, reason: err.message };
    }
  }

  buildOrder({ sellAmount, buyAmount, appData }) {
    return {
      sellToken: this.sellToken,
      buyToken: this.buyToken,
      receiver: this.receiver,
      sellAmount: sellAmount.toString(),
      buyAmount: buyAmount.toString(),
      validTo: Math.floor(this.now() / 1000) + this.ttlSeconds,
      appData,
      feeAmount: LIMIT_ORDER_FEE.toString(),
      kind: 'sell',
      partiallyFillable: this.partiallyFillable,
      sellTokenBalance: 'erc20',
      buyTokenBalance: this.buyToken === ETH_SENTINEL ? 'erc20' : 'erc20'
    };
  }

  async placeRung(rung, appData) {
    const order = this.buildOrder({
      sellAmount: rung.sellAmount,
      buyAmount: rung.buyAmount,
      appData: appData.signed
    });

    const signature = await signOrder(this.wallet, order, this.chainId);
    const expectedUid = computeOrderUid(order, this.wallet.address, this.chainId);

    const uid = await this.client.placeOrder({
      ...order,
      appData: appData.posted,
      ...(appData.posted !== appData.signed ? { appDataHash: appData.signed } : {}),
      signingScheme: 'eip712',
      signature,
      from: this.wallet.address
    });

    const returnedUid = typeof uid === 'string' ? uid : uid?.uid;
    // The UID is derived from the signed digest, so a mismatch means the orderbook
    // stored something other than what was signed.
    if (returnedUid && returnedUid.toLowerCase() !== expectedUid.toLowerCase()) {
      this.logger.error(
        `UID mismatch for ${rung.label}: expected ${expectedUid}, API returned ${returnedUid}`
      );
    }

    const record = { rung: rung.label, uid: returnedUid ?? expectedUid, order, expectedUid };
    this.placed.push(record);
    return record;
  }

  /** Approve once, then rest every rung on the book. */
  async placeLadder({ totalTokens, entryWei, ladder }) {
    const rungs = buildRungs({ totalTokens, entryWei, ladder });
    const readiness = await this.readiness(totalTokens);
    if (!readiness.routable) {
      throw new Error(`CoW cannot route this token yet: ${readiness.reason}`);
    }

    const needed = rungs.reduce((sum, r) => sum + r.sellAmount, 0n);
    await ensureAllowance({
      token: this.token, wallet: this.wallet, amount: needed, logger: this.logger
    });

    const appData = resolveAppData(readiness.quote);
    const results = [];
    for (const rung of rungs) {
      results.push(await this.placeRung(rung, appData));
    }
    return { rungs, placed: results, readiness };
  }

  async status() {
    const out = [];
    for (const record of this.placed) {
      try {
        const order = await this.client.getOrder(record.uid);
        out.push({
          rung: record.rung, uid: record.uid,
          status: order?.status,
          executedSellAmount: order?.executedSellAmount,
          executedBuyAmount: order?.executedBuyAmount
        });
      } catch (err) {
        out.push({ rung: record.rung, uid: record.uid, error: err.message });
      }
    }
    return out;
  }

  cancelAll() {
    return this.client.cancelOrders(this.placed.map((p) => p.uid), this.wallet);
  }
}
