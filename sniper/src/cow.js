import { ethers } from 'ethers';

// GPv2 contracts. These are deployed at the same addresses on every chain CoW supports,
// and match execution/cow_swap.py in this repo.
export const COW_SETTLEMENT = ethers.getAddress('0x9008D19f58AAbD9eD0D60971565AA8510560ab41');
export const COW_VAULT_RELAYER = ethers.getAddress('0xC92E8bdf79f0507f65a392b0ab4667716BFE0110');

export const COW_NETWORKS = {
  1: 'mainnet',
  100: 'xdai',
  8453: 'base',
  42161: 'arbitrum_one',
  11155111: 'sepolia'
};

/** Sentinel buyToken for receiving native ETH instead of WETH. */
export const ETH_SENTINEL = ethers.getAddress('0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE');

// The EIP-712 Order struct settled by GPv2Settlement. Field order is consensus-critical:
// it defines the type hash, so reordering silently produces signatures the solvers reject.
export const ORDER_TYPES = {
  Order: [
    { name: 'sellToken', type: 'address' },
    { name: 'buyToken', type: 'address' },
    { name: 'receiver', type: 'address' },
    { name: 'sellAmount', type: 'uint256' },
    { name: 'buyAmount', type: 'uint256' },
    { name: 'validTo', type: 'uint32' },
    { name: 'appData', type: 'bytes32' },
    { name: 'feeAmount', type: 'uint256' },
    { name: 'kind', type: 'string' },
    { name: 'partiallyFillable', type: 'bool' },
    { name: 'sellTokenBalance', type: 'string' },
    { name: 'buyTokenBalance', type: 'string' }
  ]
};

export function cowDomain(chainId) {
  return {
    name: 'Gnosis Protocol',
    version: 'v2',
    chainId: Number(chainId),
    verifyingContract: COW_SETTLEMENT
  };
}

export const orderDigest = (order, chainId) =>
  ethers.TypedDataEncoder.hash(cowDomain(chainId), ORDER_TYPES, order);

/**
 * Order UID = digest (32 bytes) || owner (20 bytes) || validTo (4 bytes, big endian).
 * Computed locally so a UID returned by the API can be cross-checked rather than trusted.
 */
export function computeOrderUid(order, owner, chainId) {
  return ethers.concat([
    orderDigest(order, chainId),
    ethers.getAddress(owner),
    ethers.toBeHex(BigInt(order.validTo), 4)
  ]);
}

export const signOrder = (wallet, order, chainId) =>
  wallet.signTypedData(cowDomain(chainId), ORDER_TYPES, order);

/**
 * Fee policy for a resting limit order.
 *
 * A quote's feeAmount is priced for immediate execution, so it is meaningless for a rung
 * that may not fill for days. CoW's limit-order convention is a zero fee taken from
 * surplus at settlement instead.
 *
 * THIS IS THE FIRST THING TO CHECK against the live API -- see README "Unverified
 * against the live API".
 */
export const LIMIT_ORDER_FEE = 0n;

export class CowError extends Error {
  constructor(message, { status, body } = {}) {
    super(message);
    this.name = 'CowError';
    this.status = status;
    this.body = body;
  }
}

export class CowClient {
  constructor({ chainId = 8453, baseUrl = 'https://api.cow.fi', fetchImpl = fetch, timeoutMs = 20_000 } = {}) {
    const network = COW_NETWORKS[Number(chainId)];
    if (!network) throw new Error(`CoW does not support chain ${chainId}`);
    this.chainId = Number(chainId);
    this.network = network;
    this.api = `${String(baseUrl).replace(/\/$/, '')}/${network}/api/v1`;
    this.fetchImpl = fetchImpl;
    this.timeoutMs = timeoutMs;
  }

  async request(path, { method = 'GET', body } = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    let response;
    try {
      response = await this.fetchImpl(`${this.api}${path}`, {
        method,
        headers: { accept: 'application/json', 'content-type': 'application/json' },
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: controller.signal
      });
    } catch (err) {
      throw new CowError(`CoW request failed for ${path}: ${err.message}`);
    } finally {
      clearTimeout(timer);
    }

    const text = await response.text();
    let parsed;
    try { parsed = text ? JSON.parse(text) : null; } catch { parsed = text; }

    if (!response.ok) {
      const detail = typeof parsed === 'object' && parsed
        ? (parsed.description ?? parsed.errorType ?? JSON.stringify(parsed))
        : String(parsed).slice(0, 300);
      throw new CowError(`CoW API ${response.status} for ${path}: ${detail}`, {
        status: response.status, body: parsed
      });
    }
    return parsed;
  }

  /** Price discovery. Also the cheapest way to learn whether solvers can route the token. */
  async quoteSell({ sellToken, buyToken, sellAmountBeforeFee, from, receiver }) {
    const response = await this.request('/quote', {
      method: 'POST',
      body: {
        sellToken, buyToken,
        sellAmountBeforeFee: String(sellAmountBeforeFee),
        kind: 'sell',
        from,
        ...(receiver ? { receiver } : {}),
        partiallyFillable: false,
        sellTokenBalance: 'erc20',
        buyTokenBalance: 'erc20'
      }
    });
    if (!response?.quote) throw new CowError('CoW quote response missing "quote"');
    return response;
  }

  async placeOrder(signedOrder) {
    return this.request('/orders', { method: 'POST', body: signedOrder });
  }

  async getOrder(uid) {
    return this.request(`/orders/${uid}`);
  }

  /** Soft-cancel a resting order by signing its UID. */
  async cancelOrders(uids, wallet) {
    const signature = await wallet.signTypedData(
      cowDomain(this.chainId),
      { OrderCancellations: [{ name: 'orderUids', type: 'bytes[]' }] },
      { orderUids: uids }
    );
    return this.request('/orders', {
      method: 'DELETE',
      body: { orderUids: uids, signature, signingScheme: 'eip712' }
    });
  }
}

/**
 * CoW pulls the sell token via the vault relayer, not the settlement contract.
 * Approving the settlement contract is the classic integration mistake: orders are
 * accepted by the API and then never settle.
 */
export async function ensureAllowance({ token, wallet, amount, logger = console }) {
  const current = await token.allowance(wallet.address, COW_VAULT_RELAYER);
  if (current >= amount) return { alreadyApproved: true, allowance: current };

  logger.log(`Approving CoW vault relayer for ${amount} (current allowance ${current})`);
  const tx = await token.approve(COW_VAULT_RELAYER, ethers.MaxUint256);
  const receipt = await tx.wait();
  if (receipt?.status !== 1) throw new Error('Vault relayer approval reverted');
  return { alreadyApproved: false, txHash: tx.hash };
}
