# Base launch sniper

Watches Base for the moment a token's WETH pool becomes tradable, then buys once —
on whichever of Uniswap V3 or Aerodrome quotes better.

```
npm install
cp .env.example .env      # fill in BASE_WSS and PRIVATE_KEY
npm test                  # 30 tests, no network required
npm run preflight         # read-only check against real Base, sends nothing
npm start                 # arm the watcher (LIVE=false until you set it)
```

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
scripts/preflight.js  read-only CLI check against real Base
test/mock-chain.js    a Base node over HTTP and websocket, for tests
```

## Tests

`npm test` runs 30 tests against a mock Base node that speaks real JSON-RPC, so calls go
through genuine ethers ABI encoding — a struct field in the wrong order fails in tests
exactly as it would on Base. Broadcasts are recorded only from
`eth_sendRawTransaction`, decoded from the signed transaction, so assertions that
"nothing was sent" mean nothing reached the network rather than nothing was simulated.

No network access and no funded wallet are required.
