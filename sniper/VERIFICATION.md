# Contract verification

Every address this bot talks to, what it was checked against, and how strong that check
is. Re-run the checks yourself before committing funds — this file records what was done,
not a guarantee.

Confidence levels used below:

- **A — read on chain.** The contract was called on Base mainnet and answered as itself.
- **B — explorer label.** Corroborated by a block explorer's own contract label.
- **C — asserted.** Believed correct, not independently confirmed here.

Re-run everything here with `npm run verify` (needs a Base RPC). Last run: block
51053833, all checks passed.

## Protocol contracts

Each contract was asked something only it can answer, and wherever possible made to
vouch for another — cross-references are far harder to fake than a label.

| Contract | Address | Level | Answered on chain |
|---|---|---|---|
| WETH (Base) | `0x4200000000000000000000000000000000000006` | A | `symbol() = WETH`, `decimals() = 18` |
| UniswapV3Factory | `0x33128a8fC17869897dcE68Ed026d694621f6FDfD` | A | All four fee tiers enabled with the canonical tick spacings (1/10/60/200) |
| QuoterV2 | `0x3d4e44Eb1374240CE5F1B871ab261CD16335B76a` | A | **`factory()` returns the factory above**; `WETH9()` returns WETH; a live WETH→USDC quote decodes |
| SwapRouter02 | `0x2626664c2603336E57B271c5C0b26F421741e481` | A | **`factory()` returns the factory above**; `WETH9()` returns WETH |
| Aerodrome PoolFactory | `0x420DD381b31aEf6683db6B902084cB0FFECe40Da` | A | `allPoolsLength()` returns 29,146 |
| Aerodrome Router | `0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43` | A | **`defaultFactory()` returns the factory above**; `weth()` returns WETH |
| GPv2Settlement | `0x9008D19f58AAbD9eD0D60971565AA8510560ab41` | A | **`vaultRelayer()` returns the relayer below**; `domainSeparator()` matches ours exactly |
| GPv2VaultRelayer | `0xC92E8bdf79f0507f65a392b0ab4667716BFE0110` | A | Has code; named by the settlement contract itself |
| Chainlink ETH/USD | `0x71041dddad3595F9CEd3DcCFBe3D1F4b0a16Bb70` | A | **`description() = "ETH / USD"`**, live price inside the sanity band |

The Chainlink feed was the weakest item at B−. `description()` returning the literal
string `ETH / USD` settles it. The runtime guards stay regardless: `src/pricing.js`
rejects a stale round, a non-positive answer, or a price outside $100–$100,000, and
preflight prints the price it used.

### The CoW domain separator is the strongest single result

`GPv2Settlement.domainSeparator()` read from the deployed contract on Base returns

```
0xd72ffa789b6fae41254d0b5a13e6e1e92ed947ec6a251edf1cf0b6c02c257b4b
```

which is byte-for-byte what `cowDomain(8453)` produces here. Every order this bot signs
commits to that separator, so the signing path is now verified against the contract that
will actually validate it — not against documentation, and not against our own mock.

## Interface identity — level A, enforced by tests

A function selector is `keccak(signature)[0:4]`, and the signature includes every struct
field's type in order. Pinning selectors therefore pins the exact calldata layout the
deployed contract expects: reorder, retype, add or drop a field and the selector moves.
`test/contracts.test.js` asserts all of these.

| Call | Selector | Why it matters |
|---|---|---|
| `getPool(address,address,uint24)` | `0x1698ee82` | Uniswap factory |
| `quoteExactInputSingle((address,address,uint256,uint24,uint160))` | `0xc6a5026a` | **QuoterV2.** V1 is `0xf7729d43` and is not deployed on Base |
| `exactInputSingle((address,address,uint24,address,uint256,uint256,uint160))` | `0x04e45aaf` | **SwapRouter02**, no deadline field. The original router is `0x414bf389` and reverts here |
| `getPool(address,address,bool)` | `0x79bc57d5` | Aerodrome — stable flag, not a fee tier |
| `getAmountsOut(uint256,(address,address,bool,address)[])` | `0x5509a1ac` | Aerodrome route tuple order |
| `swapExactETHForTokens(uint256,(...)[],address,uint256)` | `0x903638a4` | Aerodrome swap |
| `latestRoundData()` | `0xfeaf968c` | Chainlink aggregator |
| ERC-20 `balanceOf` / `approve` / `allowance` | `0x70a08231` / `0x095ea7b3` / `0xdd62ed3e` | Canonical |

## CoW order signing — level A

Read directly from `cowprotocol/contracts`, `src/contracts/libraries/GPv2Order.sol`:

```
Order(address sellToken,address buyToken,address receiver,uint256 sellAmount,
uint256 buyAmount,uint32 validTo,bytes32 appData,uint256 feeAmount,string kind,
bool partiallyFillable,string sellTokenBalance,string buyTokenBalance)
```

`TYPE_HASH = 0xd5a25ba2e97094ad7d83dc28a6572da797d6b3e7fc6663bd93efb789fc17e489`

Our `ORDER_TYPES` in `src/cow.js` produces that string character-for-character and hashes
to that exact constant. This is the strongest verification in the repo: it confirms the
struct layout every signature commits to, without needing the API.

The Base domain separator is pinned at
`0xd72ffa789b6fae41254d0b5a13e6e1e92ed947ec6a251edf1cf0b6c02c257b4b`.

### The live orderbook — both open questions answered

`npm run verify:cow` posts one order from a throwaway wallet holding nothing. It cannot
settle; the point is the rejection *reason*.

- `api.cow.fi/base/api/v1` **is live** and served (`/version` responds).
- A WETH→USDC quote returns and matches the Uniswap quote to within 0.01%.
- `appData` comes back as a bare `bytes32` zero hash, the shape `resolveAppData` handles.
- The zero-fee limit order was rejected with **`InsufficientBalance`** — *not* a fee or
  signature error. Signature validation, the struct, and `LIMIT_ORDER_FEE = 0` were all
  accepted; only the empty wallet failed. **The zero-fee convention is correct.**

## Target token — `0xB095274743941e953c746F9C228DA9c18Bb6ec29`

Corroborated as the official $LAPTOP contract by the explorer's own token page
("Hunter Biden's Laptop") and by multiple independent write-ups quoting the same address.
Launch stated as 9 September 2026 on Base, 1B supply.

From the project's published disclosures and audit:

- It is a **LayerZero OFT** (omnichain fungible token) with burn-and-mint bridging, on top
  of a standard ERC-20.
- Disclosures state **no mint, pause, freeze/blacklist or upgradeability**, and no
  transfer restrictions at the contract level.
- Audited by Hacken (report dated April 2026): no critical, high or medium findings; one
  low and three informational, stated as fixed.
- A widely shared post flagged a **`preCrime`** function as suspicious. `PreCrime` is a
  standard LayerZero security module for simulating cross-chain messages, consistent with
  an OFT deployment — not a transfer backdoor. Worth knowing rather than alarming.

Read from chain directly (level A): `symbol() = LAPTOP`, `decimals() = 18`,
`totalSupply() = 1,000,000,000`. The contract has code and answers as an ERC-20.

The audit and disclosure claims above are **not** from chain — they come from the
project's own documents and an audit they commissioned. They lower honeypot risk without
eliminating it. Keep `REQUIRE_SELL_PATH=true`.

### Pre-launch pool state

As of block 51053997 there is already a **1% fee-tier pool** at
`0xA321D950082166d11DB11cFBd6e32A91e6144Ff0` holding **0.02 WETH** — dust, seeded ahead
of launch. Its quoter call *reverts* ("Unexpected error"), which is what an uninitialized
pool does.

The bot handles this correctly, verified against live Base: discovery finds the pool,
records `below MIN_WETH_LIQ`, and returns `no-liquidity` rather than trading into it.
Both behaviours — the dust pool and the reverting quote — are captured in
`test/fixtures/base-0xb095…ec29.json` and replayed in CI.

Worth knowing: a sniper without a liquidity floor would try to trade into that pool.

## What is still not verified

- **No transaction has been sent.** Every check above is a read or a rejected order.
- **The buy has never executed**, on a fork or otherwise. `npm run fork-test` needs
  foundry, which is not installed here.
- **Whether the bot wins a launch race** — not modelled anywhere, and not knowable from
  a test.
- **Behaviour at launch**, when the real pool is funded. The recorded fixture captures
  the pre-launch state only; re-record after launch to replay the live pool.
