import os, json, streamlit as st
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd

# ─── Load keys ────────────────────────────────────────────────────────────────
SECRETS_PATH = os.path.join(os.path.dirname(__file__), "secrets/secrets.json")
with open(SECRETS_PATH) as f:
    creds = json.load(f)
client = Client(creds["BINANCE_API_KEY"], creds["BINANCE_API_SECRET"])

# ─── Helpers ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def get_balances():
    info = client.get_account()
    df = pd.DataFrame(info["balances"])
    df[["free","locked"]] = df[["free","locked"]].astype(float)
    df["total"] = df["free"] + df["locked"]
    return df[df["total"]>0].set_index("asset")[["free","locked","total"]]

def place_buy(symbol: str, usdt_amt: float):
    try:
        return client.order_market_buy(symbol=symbol, quoteOrderQty=usdt_amt)
    except BinanceAPIException as e:
        return {"error": e.message}

def place_sell(symbol: str, usdt_amt: float):
    try:
        # get current price
        ticker = client.get_symbol_ticker(symbol=symbol)
        price = float(ticker["price"])
        qty = usdt_amt / price
        # Binance requires correct step precision; truncate to 6 decimals for most coins
        qty = float(f"{qty:.6f}")
        return client.order_market_sell(symbol=symbol, quantity=qty)
    except BinanceAPIException as e:
        return {"error": e.message}

def sell_all():
    bal = get_balances()
    outs = {}
    for asset,row in bal.iterrows():
        if asset == creds["Base Currency"]:
            continue
        free = row["free"]
        sym  = asset + creds["Base Currency"]
        try:
            resp = client.order_market_sell(symbol=sym, quantity=round(free,6))
            outs[sym] = resp
        except BinanceAPIException as e:
            outs[sym] = {"error": e.message}
    return outs

# ─── Streamlit UI ────────────────────────────────────────────────────────────
st.title("📈 CryptoXQuant Trading")

st.subheader("Balances")
st.dataframe(get_balances())

st.markdown("---")
st.subheader("Market Order by USDT Amount")

col1, col2 = st.columns(2)
with col1:
    symbol = st.text_input("Symbol (e.g. BTCUSDT)", value=f"BTC{creds['Base Currency']}").upper()
    usdt_amt = st.number_input(f"Amount in {creds['Base Currency']}", min_value=0.0, step=1.0, format="%.2f")

with col2:
    if st.button("Buy"):
        result = place_buy(symbol, usdt_amt)
        st.json(result)
    if st.button("Sell"):
        result = place_sell(symbol, usdt_amt)
        st.json(result)

st.markdown("---")
st.subheader("Close All Positions")
if st.button("Sell All"):
    results = sell_all()
    st.json(results)