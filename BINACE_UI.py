import os
import json
import streamlit as st
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd

# ─── 1) Load your API keys ───────────────────────────────────────────────────────
SECRETS_PATH = 'secrets/secrets.json'
with open(SECRETS_PATH) as f:
    creds = json.load(f)
client = Client(creds["BINANCE_API_KEY"], creds["BINANCE_API_SECRET"])

# ─── 2) Helper functions ───────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def get_balances():
    """Fetch all asset balances (free + locked) and return non-zero ones."""
    info = client.get_account()
    df = pd.DataFrame(info["balances"])
    df[["free","locked"]] = df[["free","locked"]].astype(float)
    df["total"] = df["free"] + df["locked"]
    df = df[df["total"]>0]
    return df.set_index("asset")[["free","locked","total"]]

def place_order(symbol: str, side: str, qty: float):
    """Place a market buy or sell, return Binance response or error."""
    try:
        if side == "BUY":
            return client.order_market_buy(symbol=symbol, quantity=qty)
        else:
            return client.order_market_sell(symbol=symbol, quantity=qty)
    except BinanceAPIException as e:
        return {"error": e.message}

def sell_all_positions():
    """For each non-USDT balance, sell everything into USDT."""
    bal = get_balances()
    outs = {}
    for asset, row in bal.iterrows():
        if asset == creds["Base Currency"]:
            continue
        free = row["free"]
        symbol = asset + creds["Base Currency"]
        try:
            resp = client.order_market_sell(symbol=symbol, quantity=round(free,8))
            outs[symbol] = resp
        except BinanceAPIException as e:
            outs[symbol] = {"error": e.message}
    return outs

# ─── 3) Streamlit UI ───────────────────────────────────────────────────────────
st.set_page_config(page_title="CryptoXQuant Trading", layout="wide")
st.title("📈 CryptoXQuant Spot Trading")

# Show balances
st.subheader("Your Binance Balances")
balances = get_balances()
st.dataframe(balances)

st.markdown("---")

# Trade panel
st.subheader("Place a Market Order")
col1, col2 = st.columns(2)

with col1:
    symbol = st.text_input("Ticker symbol (e.g. BTCUSDT)", value=f"BTC{creds['Base Currency']}")
    qty    = st.number_input("Quantity to trade", min_value=0.0, step=0.0001, format="%.8f")

with col2:
    if st.button("Buy"):
        result = place_order(symbol.upper(), "BUY", qty)
        st.write(result)
    if st.button("Sell"):
        result = place_order(symbol.upper(), "SELL", qty)
        st.write(result)

st.markdown("---")

# Sell all button
st.subheader("Close All Positions")
if st.button("Sell All"):
    results = sell_all_positions()
    st.json(results)