# Base launch sniper

Watches Base for the moment a token's WETH pool becomes tradable, then buys once —
on whichever of Uniswap V3 or Aerodrome quotes better.

```
npm install
cp .env.example .env      # fill in BASE_WSS and PRIVATE_KEY
npm test                  # 30 tests, no network required
npm run preflight         # read-only check against real Base, sends nothing
npm start                 # arm the watcher (LIVE=false until you set it)

npm run demo:exit         # offline walkthrough of the exit path, no network
npm run exit -- --plan    # show the take-profit ladder, touch nothing
```

## How much of this is actually verified

Worth being blunt, because the test count is misleading on its own.

| Layer | What it proves | What it does not |
|---|---|---|
| `test/contracts.test.js` | **Interface identity.** Every selector, the CoW order type hash and the domain separator are pinned to independently verified constants. A selector is a hash over the full signature including struct field order, so this pins the exact calldata the deployed contracts expect | That the addresses are the right *contracts* — see VERIFICATION.md for how each was checked |
| `npm test` (84 tests) | The code is self-consistent: guard ordering, exact bigint math, real ABI encode/decode, EIP-712 signatures that recover, reconnect behaviour | Little about live Base. The hand-written mocks encode this repo's *assumptions*; where an assumption is wrong, mock and code are wrong together and those tests still pass |
| `npm run record-fixtures` then `npm test` | Replays **real recorded Base responses** through the real decode path. Unlike the mocks, these bytes cannot agree with a mistake in `src/` | Only the read path, at the block it was recorded |
| `npm run preflight` | Real Base answers: chain, token, wallet funding, live pool state, the ETH/USD price actually used | Nothing about executing a trade |
| `npm run fork-test` | The **real** Uniswap/Aerodrome contracts accept the calls and the buy fills, at real reserves | Nothing about the target token before it launches, or about winning the race |
| a small live buy | The whole path, with real money | Only what that size and moment showed |

The hand-written mocks are the weak link by construction: `src/cow.js` and
`test/mock-cow.js` were written together, so a misread of the CoW API breaks both
identically. Two things now push back on that:

- **`test/contracts.test.js`** pins selectors and the CoW `TYPE_HASH` to constants taken
  from the protocols themselves. These cannot drift along with a mistake in `src/`.
- **`npm run record-fixtures`** captures real Base responses to `test/fixtures/`, which
  the replay tests then feed through the real decode path. Commit the fixture and CI
  exercises genuine wire data with no network of its own.

```
BASE_RPC=https://... TOKEN=0x<a token with liquidity> npm run record-fixtures
npm test        # replay tests light up automatically once a fixture exists
```

See **VERIFICATION.md** for every address, what it was checked against, and how strong
each check is.

`npm run fork-test` is the cheapest real evidence available and the one worth running
before committing size. It needs foundry and a Base RPC:

```
anvil --fork-url https://mainnet.base.org --port 8545
FORK_RPC=http://127.0.0.1:8545 TOKEN=0x<a token that already trades> npm run fork-test
```

Point `TOKEN` at something with existing liquidity. Before launch the real target has no
pool, so a fork of today only proves discovery correctly finds nothing — which is worth
knowing, but does not exercise the buy.

## Safety model

The watcher will not send a transaction unless **all** of these hold:

| Gate | Behaviour |
|---|---|
| `LIVE` | `false` (default) observes and prints only |
| `EARLIEST_BUY` | no buy attempted before this instant |
| `MIN_WETH_LIQ` | pool must hold at least this much WETH |
| `MIN_TOKENS_OUT` | absolute floor on tokens received — **required** when `LIVE=true` |
| `SLIPPAGE_BPS` | relative floor against the quote just taken |
| `REQUIRE_SELL_PATH` | round trip must quote, and return >50% of input |
| `MAX_PRICE_IMPACT_BPS` | refuse a fill worse than this vs. the pool's undisturbed price |
| staleness | quote older than `MAX_SCAN_STALENESS_MS` is discarded, not traded on |
| `staticCall` | full simulation from the buying wallet before anything is broadcast |

`amountOutMinimum` is `max(slippage floor, MIN_TOKENS_OUT)` — the stricter of the two
always wins.

## Why `MIN_TOKENS_OUT` is mandatory in live mode

Slippage protection is relative to a quote taken moments earlier, so it happily lets you
buy into a pool that opened at any price at all — it only stops the price moving away
from that opening. `MIN_TOKENS_OUT` is the absolute floor, expressed in whole tokens for
`BUY_ETH` worth of buying, and it is the only setting here that encodes what you
actually think the token is worth. Startup refuses to arm in live mode without it.

## What the sell-side probe does and does not prove

Before buying, the watcher quotes the round trip (`BUY_ETH` → token → WETH) and refuses
to trade if there is no reverse route, or if the round trip returns less than half the
input. That catches a missing sell side and a punitive transfer tax.

It does **not** prove the token is sellable. A contract can allow the quote and block the
transfer, or whitelist addresses, or switch behaviour after launch. Nothing checkable
before the buy rules that out. Size the position accordingly.

## Sizing, and why it changes the other settings

`MIN_WETH_LIQ` is a threshold on the pool, but what matters is your size *relative* to
the pool. The defaults suit a 0.01 ETH buy. Scaling to four figures without touching them
means trading into a pool a tenth as deep as you need:

| Buy | Pool WETH | Approx. price impact |
|---|---|---|
| 0.01 ETH | 5 | ~0.1% |
| 0.227 ETH (~$1k) | 5 | ~9% |
| 0.227 ETH (~$1k) | 25 | ~1.8% |
| 0.227 ETH (~$1k) | 50 | ~0.9% |

So a four-figure buy wants `MIN_WETH_LIQ` in the 25-50 range, not 5. `MAX_PRICE_IMPACT_BPS`
is the backstop that catches this regardless: it quotes a probe at 1/100th your size,
compares the rates, and refuses the trade if the gap exceeds the cap.

`BUY_USD=1000` reads Chainlink ETH/USD on Base at arm time. The feed address is in
`src/constants.js` and is **not** verified by this repo — preflight prints the price it
read, so a wrong feed shows up as an obviously wrong number rather than a quietly
mis-sized trade. Check that line before every live run.

## Verify the token

`src/constants.js` carries the token address as a constant. Nothing in this repo can
verify that it is the official one — check it against the project's own announcement
before a live run. A wrong address here loses the whole `BUY_ETH` to an impostor
contract, and every other guard in the table above will pass while it happens.

## Operational notes

A launch watcher's job is to be awake at one specific moment, so the failure that matters
is a websocket that dies quietly: the process stays up, looks healthy, and never fires
again. A watchdog rebuilds the connection after `BLOCK_WATCHDOG_MS` of silence, and the
`bought` flag survives the rebuild so a reconnect cannot double-buy. This is covered by
tests that make a live socket go deaf mid-run.

Run it somewhere durable — a VPS or a long-lived container. It must outlive your laptop
lid and any session that gets reclaimed.

## Exit: the CoW take-profit ladder

The buy goes through the AMMs, where landing in the right block is everything. The
**sell** goes through CoW Protocol, where it is worth waiting: dumping a fresh token into
a thin pool is the textbook sandwich target, and CoW's batch auctions settle at a uniform
clearing price, so there is no ordering advantage to sell you.

On a fill the sniper writes `position.json` with what actually landed. The ladder prices
off that:

```
npm run exit -- --plan     # rungs and proceeds, nothing approved or placed
npm run exit -- --place    # approve the vault relayer, rest every rung on CoW
npm run exit -- --status    # poll fills
npm run exit -- --cancel    # cancel every resting rung
```

`EXIT_LADDER=3x:25,5x:25,10x:50` reads as: sell 25% at 3x the entry price, 25% at 5x, the
rest at 10x. Percentages are of the original position and are rejected at startup if they
exceed 100%. Each rung rests as a partially fillable limit order, so thin liquidity fills
it in pieces rather than not at all.

Three things worth knowing:

- **Approval goes to the vault relayer** (`0xC92E…0110`), not the settlement contract.
  Approving settlement is the classic CoW integration bug: the orderbook accepts your
  orders and they never settle. A test asserts the spender.
- **The ladder prices off the balance actually held**, not the quote the buy hoped for.
  If a transfer tax ate 2% on the way in, the rungs still ask for 3x of what you really
  paid per token.
- **Readiness is checked before the approval.** If no solver can route the token yet, the
  run stops with nothing approved and nothing placed, rather than leaving an allowance
  behind for a token that cannot be sold.

## Unverified against the live API

Everything here is tested against a mock orderbook that recovers the EIP-712 signer, so
the signing is self-consistent and the Order struct matches the published GPv2 layout.
What the mock cannot tell you is whether the live orderbook agrees. Before trusting real
funds to it, check two things against `api.cow.fi`:

1. **Base is live on the API** — `curl https://api.cow.fi/base/api/v1/version`. This repo
   already assumes so (`execution/cow_swap.py` maps `base: 8453`), and the GPv2 addresses
   are the same on every CoW chain, but it is one command.
2. **The limit-order fee convention.** `LIMIT_ORDER_FEE` in `src/cow.js` is `0`, on the
   basis that a quote's fee is priced for immediate execution and is meaningless for a
   rung that may rest for days. If the API rejects a zero fee, that constant is the one
   place to change.

`npm run exit -- --plan` is safe to run against the real API: it quotes nothing and
places nothing.

## Layout

```
index.js              thin runner: load config, arm the watcher, print the banner
src/config.js         env parsing; every non-network check happens here, at startup
src/constants.js      addresses and ABIs
src/wiring.js         contract construction (reads on provider, routers on wallet)
src/preflight.js      chain id, token code, wallet funding
src/venues.js         pool discovery, quoting, sell-side probe
src/sniper.js         venue selection, guards, execution
src/watcher.js        socket lifecycle, block subscription, watchdog reconnect
src/cow.js            CoW client, EIP-712 order signing, order UID, allowance
src/exit.js           ladder parsing, rung pricing, order placement and polling
scripts/preflight.js  read-only CLI check against real Base
scripts/exit.js       plan / place / status / cancel the ladder
scripts/demo-exit.js  offline walkthrough of the whole exit path
test/mock-chain.js    a Base node over HTTP and websocket, for tests
test/mock-cow.js      a CoW orderbook that verifies EIP-712 signatures
```

## Tests

`npm test` runs 47 tests against a mock Base node that speaks real JSON-RPC, so calls go
through genuine ethers ABI encoding — a struct field in the wrong order fails in tests
exactly as it would on Base. Broadcasts are recorded only from
`eth_sendRawTransaction`, decoded from the signed transaction, so assertions that
"nothing was sent" mean nothing reached the network rather than nothing was simulated.

The CoW tests go through a mock orderbook that **recovers the EIP-712 signer** from each
order rather than just checking the payload's shape, so a reordered struct field or a
wrong domain fails the same way solvers would reject it. One test tampers with an order
after signing and asserts the recovery catches it.

No network access and no funded wallet are required.
