# wstETH/WETH Looping Risk — Oracle & Liquidation Drivers (Aave V3)
**Date:** Feb 12, 2026  

## Summary
For a **wstETH collateral / WETH debt eMode** loop on Aave, **DEX stETH/ETH depegs and ETH/USD moves do not directly change Health Factor (HF)** because the oracle path pegs stETH/ETH at 1:1 and **ETH/USD cancels out** in this specific pair.  
**Primary risk is utilization → borrow-rate spikes → negative carry → debt growth → HF erosion (liquidation on a delay)**, plus **unwind execution costs** during stress.

---

## Findings

### 1) stETH/ETH is pegged 1:1 by oracle adapter design (not market price)
Aave’s stETH→ETH adapter returns **1 stETH = 1 ETH** and **does not read Curve/Uniswap/secondary markets**.  
Source: BGD Labs confirmation + adapter code.  
- https://governance.aave.com/t/exchange-rate-for-steth-eth-hardcoded/22693  
- https://github.com/bgd-labs/cl-synchronicity-price-adapter/blob/main/src/contracts/StETHtoETHSynchronicityPriceAdapter.sol  

**Implication:** A Curve/Uni stETH discount does **not** directly lower HF for wstETH/WETH eMode loops.

### 2) wstETH is priced from Lido’s on-chain exchange rate (not DEX)
Aave’s wstETH pricing uses **Lido’s `getPooledEthByShares()`** (protocol exchange rate) rather than market price.  
- https://github.com/bgd-labs/cl-synchronicity-price-adapter/blob/main/src/contracts/WstETHSynchronicityPriceAdapter.sol  
- On-chain deployed CAPO adapter (verified): https://etherscan.io/address/0xe1D97bF61901B075E9626c8A2340a7De385861Ef  

**CAPO nuance:** upward growth is capped (9.68%/yr); no downward cap.

### 3) ETH/USD cancels out of HF for the *single-collateral wstETH / single-debt WETH* loop
For this primitive, both collateral and debt share the **same ETH/USD leg**, so it cancels in HF. HF depends on:
- Lido exchange rate (`getPooledEthByShares`)  
- eMode LT (governance-controlled)  
- collateral/debt ratio (including interest index growth)

HF code:  
- https://github.com/aave/aave-v3-core/blob/master/contracts/protocol/libraries/logic/GenericLogic.sol  

**Scope note:** This cancellation is exact for the wstETH/WETH single-pair loop; it does not necessarily hold at the full-account level if multiple assets/oracle paths are involved.

---

## Liquidation triggers for wstETH/WETH eMode (what can actually push HF < 1)
Even though DEX depegs don’t hit the oracle, **liquidation is still possible** via:
1) **Borrowed amount increases (interest accrual)** → WETH debt grows via borrow index  
2) **Lido exchange rate drops** (slashing/penalties reduce pooled ETH per share)  
3) **Governance/risk-steward parameter changes** (eMode LT, IR curve, CAPO config)

Aave explicitly lists liquidation pathways as “collateral decreases OR borrowed amount increases”:  
- https://aave.com/help/borrowing/liquidations  

Chaos Labs explicitly flags **accrued interest** as a liquidation driver for LST–WETH positions:  
- https://governance.aave.com/t/risk-stewards-wsteth-weth-emode-update-ethereum-arbitrum-base-instances/21333  

---

## Stress transmission we should model (what replaces “depeg causes liquidation”)
Stress channel for loopers is primarily:

**ETH down → liquidations elsewhere → WETH removed from pool → utilization ↑ → borrow APR ↑ → negative carry + debt growth → unwind pressure + eventual HF erosion**.

Recent example: Chaos Labs (Feb 7, 2026) proposed emergency WETH IR curve adjustments due to “sharp ETH supply contractions” and elevated utilization.  
- https://governance.aave.com/t/chaos-labs-risk-stewards-adjust-weth-interest-rate-curve-on-aave-v3-07-02-26/24018  

**Implication for our dashboard/model:** model **utilization dynamics** as the key state variable; treat depeg as an **unwind cost variable**, not an oracle/liquidation driver.

---

## Potential Changes to our model
- **VaR driver:** shift from “stETH depeg drives HF” → **carry/rate spike risk + unwind slippage + rare slashing**.  
- **Depeg modeling stays**, but only for **execution cost under deleveraging**, not HF.  
- **Next-day APY confidence interval** should be derived from **utilization distribution → rate curve** (kink makes the tail extremely impactful).  
- Always treat LT + IR params as **governance-controlled state**, pulled from chain at the relevant block/time.

---

## Key links 
- BGD: stETH/ETH pegged 1:1 (adapter rationale): https://governance.aave.com/t/exchange-rate-for-steth-eth-hardcoded/22693  
- stETH→ETH adapter code: https://github.com/bgd-labs/cl-synchronicity-price-adapter/blob/main/src/contracts/StETHtoETHSynchronicityPriceAdapter.sol  
- wstETH adapter code: https://github.com/bgd-labs/cl-synchronicity-price-adapter/blob/main/src/contracts/WstETHSynchronicityPriceAdapter.sol  
- Deployed wstETH CAPO adapter (verified): https://etherscan.io/address/0xe1D97bF61901B075E9626c8A2340a7De385861Ef  
- HF computation (GenericLogic): https://github.com/aave/aave-v3-core/blob/master/contracts/protocol/libraries/logic/GenericLogic.sol  
- Aave liquidation explainer (“borrowed amount increases”): https://aave.com/help/borrowing/liquidations  
- Chaos Labs: interest accrual + LST–WETH liquidation discussion: https://governance.aave.com/t/risk-stewards-wsteth-weth-emode-update-ethereum-arbitrum-base-instances/21333  
- Chaos Labs: Feb 2026 WETH IR curve changes during stress: https://governance.aave.com/t/chaos-labs-risk-stewards-adjust-weth-interest-rate-curve-on-aave-v3-07-02-26/24018  

