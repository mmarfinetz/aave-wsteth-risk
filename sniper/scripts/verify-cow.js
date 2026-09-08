#!/usr/bin/env node
/**
 * Verify the CoW orderbook contract against the live API.
 *
 * Two questions the mocks could not answer, because this repo wrote both the client and
 * the mock: is Base actually served, and does the book accept a zero-fee limit order?
 *
 * The order posted here is signed by a throwaway wallet that holds nothing and has no
 * approval, so it cannot settle. That is deliberate: the rejection *reason* is the
 * answer. A balance complaint means the format was accepted; a fee complaint means
 * LIMIT_ORDER_FEE is wrong.
 */
import { ethers } from 'ethers';
import { CowClient, signOrder, computeOrderUid, LIMIT_ORDER_FEE } from '../src/cow.js';
import { resolveAppData } from '../src/exit.js';
import { WETH } from '../src/constants.js';
import './_proxy.js';

const USDC = ethers.getAddress('0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913');
const ok = (m) => console.log(`  ok    ${m}`);
const bad = (m) => console.log(`  FAIL  ${m}`);
const info = (m) => console.log(`  ..    ${m}`);

// Route the API through the sandbox proxy when one is configured. Node's global fetch
// carries its own internal undici, so a ProxyAgent from the npm package is ignored
// there -- undici's own fetch has to be used for the dispatcher to take effect.
const PROXY = process.env.HTTPS_PROXY ?? process.env.https_proxy;
let fetchImpl = fetch;
if (PROXY) {
  const { fetch: undiciFetch, ProxyAgent } = await import('undici');
  const dispatcher = new ProxyAgent(PROXY);
  fetchImpl = (url, opts = {}) => undiciFetch(url, { ...opts, dispatcher });
}

const client = new CowClient({ chainId: 8453, fetchImpl });
const wallet = ethers.Wallet.createRandom();
console.log(`CoW API: ${client.api}`);
console.log(`throwaway signer: ${wallet.address} (unfunded, no approval)\n`);

try {
  console.log('Reachability');
  const version = await client.request('/version').catch((e) => ({ error: e.message }));
  if (version?.error) bad(`/version: ${version.error}`);
  else ok(`/version responds: ${JSON.stringify(version)}`);

  console.log('\nQuoting a liquid pair (WETH -> USDC)');
  let quote;
  try {
    quote = await client.quoteSell({
      sellToken: WETH, buyToken: USDC,
      sellAmountBeforeFee: ethers.parseEther('1'),
      from: wallet.address
    });
    ok(`quote returned, buyAmount ${ethers.formatUnits(quote.quote.buyAmount, 6)} USDC`);
    ok(`quote feeAmount ${ethers.formatEther(quote.quote.feeAmount)} WETH ` +
       '(priced for immediate execution)');
    info(`validTo ${quote.quote.validTo}, kind ${quote.quote.kind}`);
    info(`appData in response: ${JSON.stringify(quote.quote.appData).slice(0, 80)}`);
    info(`appDataHash present: ${quote.appDataHash ?? quote.quote.appDataHash ?? '(no)'}`);
    // Confirm our parser handles whatever shape the live API actually returned.
    const resolved = resolveAppData(quote);
    ok(`resolveAppData -> signed ${resolved.signed.slice(0, 18)}...`);
  } catch (err) {
    bad(`quote failed: ${err.message}`);
  }

  if (quote) {
    console.log('\nZero-fee limit order acceptance');
    const order = {
      sellToken: WETH, buyToken: USDC, receiver: wallet.address,
      sellAmount: ethers.parseEther('1').toString(),
      // Ask far above market so it could never fill even if it were funded.
      buyAmount: (BigInt(quote.quote.buyAmount) * 10n).toString(),
      validTo: Math.floor(Date.now() / 1000) + 3600,
      appData: resolveAppData(quote).signed,
      feeAmount: LIMIT_ORDER_FEE.toString(),
      kind: 'sell', partiallyFillable: true,
      sellTokenBalance: 'erc20', buyTokenBalance: 'erc20'
    };
    const signature = await signOrder(wallet, order, 8453);
    const expectedUid = computeOrderUid(order, wallet.address, 8453);
    info(`locally computed UID ${expectedUid.slice(0, 22)}...`);

    try {
      const uid = await client.placeOrder({
        ...order, appData: resolveAppData(quote).posted,
        signingScheme: 'eip712', signature, from: wallet.address
      });
      ok(`order ACCEPTED, uid ${uid}`);
      const returned = typeof uid === 'string' ? uid : uid?.uid;
      if (returned && returned.toLowerCase() === expectedUid.toLowerCase()) {
        ok('returned UID matches our local derivation exactly');
      }
      info('cancelling it so nothing is left on the book...');
      await client.cancelOrders([returned], wallet).then(
        () => ok('cancelled'), (e) => info(`cancel said: ${e.message}`));
    } catch (err) {
      const body = err.body ?? {};
      const type = body.errorType ?? '(none)';
      console.log(`  ..    rejected with errorType=${type}`);
      console.log(`  ..    ${body.description ?? err.message}`);
      if (/balance|allowance|funds/i.test(`${type} ${body.description}`)) {
        ok('rejected for BALANCE/ALLOWANCE — signature, fee and struct were accepted');
        ok('zero-fee limit order format is valid on this book');
      } else if (/fee/i.test(`${type} ${body.description}`)) {
        bad('rejected on FEE — LIMIT_ORDER_FEE=0 is not accepted; change it in src/cow.js');
      } else if (/signature/i.test(`${type} ${body.description}`)) {
        bad('rejected on SIGNATURE — the EIP-712 struct or domain is wrong');
      } else {
        info('inconclusive: read the errorType above');
      }
    }
  }
} catch (err) {
  bad(err.message ?? String(err));
  process.exitCode = 1;
}
