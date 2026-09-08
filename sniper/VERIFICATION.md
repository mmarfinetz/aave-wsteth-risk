# Contract verification

Every address this bot talks to, what it was checked against, and how strong that check
is. Re-run the checks yourself before committing funds — this file records what was done,
not a guarantee.

Confidence levels used below:

- **A — protocol source.** Read directly from the protocol's own repository.
- **B — explorer label.** Corroborated by a block explorer's own contract label for that
  exact address, plus at least one other source.
- **C — asserted.** Believed correct, not independently confirmed here.

## Protocol contracts

| Contract | Address | Level | Checked against |
|---|---|---|---|
| WETH (Base) | `0x4200000000000000000000000000000000000006` | B | Base predeploy, universally documented |
| UniswapV3Factory | `0x33128a8fC17869897dcE68Ed026d694621f6FDfD` | B | Uniswap Base deployments doc + explorer |
| QuoterV2 | `0x3d4e44Eb1374240CE5F1B871ab261CD16335B76a` | B | BaseScan label "Uniswap V3: QuoterV2" |
| SwapRouter02 | `0x2626664c2603336E57B271c5C0b26F421741e481` | B | BaseScan label "Uniswap V3: Swap Router02" |
| Aerodrome PoolFactory | `0x420DD381b31aEf6683db6B902084cB0FFECe40Da` | B | BaseScan label "Aerodrome: Pool Factory" |
| Aerodrome Router | `0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43` | B | BaseScan label "Aerodrome: Router" |
| GPv2Settlement | `0x9008D19f58AAbD9eD0D60971565AA8510560ab41` | B | CoW docs + explorer; same address on every CoW chain |
| GPv2VaultRelayer | `0xC92E8bdf79f0507f65a392b0ab4667716BFE0110` | B | Explorer label + CoW docs ("only sign approvals to this address") |
| Chainlink ETH/USD | `0x71041dddad3595F9CEd3DcCFBe3D1F4b0a16Bb70` | B− | Documented as the Base ETH/USD feed; weaker corroboration than the rest |

The Chainlink feed is the softest of these. It is also the one with a built-in check:
preflight prints the price it read, and `src/pricing.js` rejects anything stale,
non-positive, or outside a $100–$100,000 band. A wrong feed shows up as an obviously
wrong number rather than a quietly mis-sized trade. **Read that line before every run.**

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

Still unconfirmed on the CoW side: whether the live orderbook accepts a **zero fee** on a
resting limit order (`LIMIT_ORDER_FEE` in `src/cow.js`), and whether Base is currently
served at `api.cow.fi/base`. Both need one live call — `npm run exit -- --plan` is safe
and would answer them.

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

**None of this was read from chain.** It comes from the project's own documents and an
audit commissioned by them. It lowers honeypot risk; it does not eliminate it, and it is
not a statement about whether the token is a good trade. Keep `REQUIRE_SELL_PATH=true`.

## What is still not verified

- Nothing here executed a transaction, or called Base or `api.cow.fi`.
- Explorer labels were read from search result titles; the explorer itself is blocked from
  this environment.
- Whether the bot wins a launch race — not modelled anywhere.

`npm run record-fixtures` and `npm run fork-test` are the two commands that convert this
from documentation into evidence. Both need network access.
