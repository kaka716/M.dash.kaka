import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# ------------------ Page Setup (mobile-friendly) ------------------
st.set_page_config(page_title="داشبورد محصول + حساسیت (۵ محصول)", layout="centered")
st.title("📊 داشبورد تعاملی محصولات + تحلیل حساسیت")
st.caption("نسخه‌ی سبک برای موبایل • داده‌ها به‌صورت مصنوعی تولید می‌شوند")

# ------------------ Synthetic Data (5 products) ------------------
@st.cache_data
def make_data(seed=123, days=180):
    rng = np.random.default_rng(seed)
    start = datetime.today().date() - timedelta(days=days)
    dates = pd.date_range(start, periods=days, freq="D")

    products = [
        {"product": "Apex Lite",     "base_price": 39.0, "cost": 21.0, "elastic": -0.9, "beta_mkt": 0.35},
        {"product": "Nova Plus",     "base_price": 59.0, "cost": 33.0, "elastic": -1.1, "beta_mkt": 0.40},
        {"product": "Orion Pro",     "base_price": 89.0, "cost": 52.0, "elastic": -1.2, "beta_mkt": 0.45},
        {"product": "Zen Mini",      "base_price": 24.0, "cost": 12.0, "elastic": -0.7, "beta_mkt": 0.30},
        {"product": "Pulse Ultra",   "base_price": 129.0,"cost": 74.0, "elastic": -1.3, "beta_mkt": 0.50},
    ]

    rows = []
    for p in products:
        mkt_base = rng.normal(1500, 400, size=days)
        week_wave = 1 + 0.15*np.sin(np.linspace(0, 6*np.pi, days))
        mkt = np.clip(mkt_base * week_wave, 200, None)

        price = p["base_price"] * (1 + rng.normal(0, 0.03, size=days))

        base_demand = 2400 / np.sqrt(p["base_price"])
        noise = rng.normal(0, 6, size=days)

        units_hat = base_demand + p["elastic"]*(price - p["base_price"]) + p["beta_mkt"]*(mkt/np.mean(mkt)) + noise
        units = np.maximum(units_hat, 0).round()

        df_p = pd.DataFrame({
            "date": dates,
            "product": p["product"],
            "price": price.round(2),
            "marketing_spend": mkt.round(0),
            "units": units.astype(int),
            "unit_cost": p["cost"],
            "elastic": p["elastic"],
            "beta_mkt": p["beta_mkt"],
        })
        df_p["revenue"] = (df_p["price"] * df_p["units"]).round(2)
        df_p["profit"] = (df_p["revenue"] - (df_p["unit_cost"]*df_p["units"] + 0.12*df_p["marketing_spend"])).round(2)
        rows.append(df_p)

    return pd.concat(rows, ignore_index=True)

df = make_data()

# ------------------ Controls ------------------
with st.expander("⚙️ تنظیمات و فیلترها", expanded=True):
    prods = sorted(df["product"].unique().tolist())
    selected_products = st.multiselect("محصولات", prods, default=prods)
    min_d, max_d = df["date"].min().date(), df["date"].max().date()
    date_range = st.date_input("بازه‌ی تاریخ", (min_d, max_d), min_value=min_d, max_value=max_d)
    metric = st.selectbox("سنجه‌ی اصلی", ["revenue", "profit", "units"])
    smoothed = st.checkbox("میانگین متحرک ۷روزه برای روند", value=True)

df_f = df[df["product"].isin(selected_products)].copy()
if isinstance(date_range, tuple) and len(date_range) == 2:
    d1, d2 = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df_f = df_f[(df_f["date"] >= d1) & (df_f["date"] <= d2)]

# ------------------ KPIs ------------------
sum_rev = float(df_f["revenue"].sum())
sum_profit = float(df_f["profit"].sum())
sum_units = int(df_f["units"].sum())
c1, c2, c3 = st.columns(3)
c1.metric("💰 درآمد", f"{sum_rev:,.0f}")
c2.metric("📈 سود", f"{sum_profit:,.0f}")
c3.metric("📦 تعداد فروش", f"{sum_units:,}")

# ------------------ Charts ------------------
tab1, tab2 = st.tabs(["📊 داشبورد", "🧪 تحلیل حساسیت"])

with tab1:
    agg = df_f.groupby("product", as_index=False).agg(
        revenue=("revenue","sum"),
        profit=("profit","sum"),
        units=("units","sum"),
        price=("price","mean")
    )
    fig_bar = px.bar(agg.sort_values(metric, ascending=False), x="product", y=metric, text=metric, title=f"{metric} بر اساس محصول")
    st.plotly_chart(fig_bar, use_container_width=True)

    ts = df_f.groupby(["date","product"], as_index=False).agg(value=(metric,"sum"))
    if smoothed:
        ts["value"] = ts.groupby("product")["value"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    fig_ts = px.line(ts, x="date", y="value", color="product", title=f"روند روزانه {metric}")
    st.plotly_chart(fig_ts, use_container_width=True)

    scat = df_f.copy()
    scat["bubble"] = scat["revenue"]
    fig_sc = px.scatter(scat, x="price", y="units", size="bubble", hover_data=["product","date"], title="قیمت vs. تعداد (حباب = درآمد)")
    st.plotly_chart(fig_sc, use_container_width=True)

with tab2:
    st.subheader("What-if: سناریوی قیمت/مارکتینگ/هزینه/کشش")
    st.caption("پارامترها را تغییر بده و اثر روی درآمد/سود را ببین.")

    price_mult    = st.slider("ضریب قیمت (همه‌ی محصولات)", 0.6, 1.4, 1.0, 0.01)
    mkt_mult      = st.slider("ضریب هزینه‌ی مارکتینگ", 0.5, 2.0, 1.0, 0.05)
    cost_mult     = st.slider("ضریب هزینه‌ی واحد تولید", 0.8, 1.5, 1.0, 0.01)
    elastic_tweak = st.slider("تغییر کشش قیمت (واحدی)", -0.5, 0.5, 0.0, 0.05)
    beta_tweak    = st.slider("تغییر حساسیت به مارکتینگ (واحدی)", -0.2, 0.2, 0.0, 0.01)

    base_rev = float(df_f["revenue"].sum())
    base_profit = float(df_f["profit"].sum())

    scen = df_f.copy()
    scen["price_scn"] = scen["price"] * price_mult
    scen["mkt_scn"] = scen["marketing_spend"] * mkt_mult
    scen["unit_cost_scn"] = scen["unit_cost"] * cost_mult
    scen["elastic_scn"] = scen["elastic"] + elastic_tweak
    scen["beta_scn"] = scen["beta_mkt"] + beta_tweak

    mkt_norm = df_f["marketing_spend"].mean() if df_f["marketing_spend"].mean() != 0 else 1.0
    units_hat = ((2400 / np.sqrt(scen["price"]))
                 + scen["elastic_scn"]*(scen["price_scn"] - scen["price"])
                 + scen["beta_scn"]*(scen["mkt_scn"]/mkt_norm))
    units_pred = np.maximum(units_hat, 0).round()

    scen["revenue_scn"] = scen["price_scn"] * units_pred
    scen["profit_scn"]  = scen["revenue_scn"] - (scen["unit_cost_scn"]*units_pred + 0.12*scen["mkt_scn"])

    scen_rev = float(scen["revenue_scn"].sum())
    scen_profit = float(scen["profit_scn"].sum())

    k1, k2 = st.columns(2)
    delta_rev = 0.0 if base_rev==0 else (scen_rev-base_rev)/base_rev*100
    delta_profit = 0.0 if base_profit==0 else (scen_profit-base_profit)/base_profit*100
    k1.metric("درآمد سناریو", f"{scen_rev:,.0f}", f"{delta_rev:+.1f}% vs. مبنا")
    k2.metric("سود سناریو", f"{scen_profit:,.0f}", f"{delta_profit:+.1f}% vs. مبنا")

    cmp_df = pd.DataFrame({
        "حالت": ["Baseline","Scenario"],
        "Revenue": [base_rev, scen_rev],
        "Profit": [base_profit, scen_profit]
    })
    st.plotly_chart(px.bar(cmp_df, x="حالت", y="Revenue", text="Revenue", title="درآمد: مبنا vs. سناریو"), use_container_width=True)
    st.plotly_chart(px.bar(cmp_df, x="حالت", y="Profit", text="Profit", title="سود: مبنا vs. سناریو"), use_container_width=True)

    def run_sum(pm=price_mult, mm=mkt_mult, cm=cost_mult, et=elastic_tweak, bt=beta_tweak):
        sc = df_f.copy()
        sc["price_scn"] = sc["price"] * pm
        sc["mkt_scn"] = sc["marketing_spend"] * mm
        sc["unit_cost_scn"] = sc["unit_cost"] * cm
        sc["elastic_scn"] = sc["elastic"] + et
        sc["beta_scn"] = sc["beta_mkt"] + bt
        mkt_norm = df_f["marketing_spend"].mean() if df_f["marketing_spend"].mean() != 0 else 1.0
        units_hat = ((2400 / np.sqrt(sc["price"]))
                     + sc["elastic_scn"]*(sc["price_scn"] - sc["price"])
                     + sc["beta_scn"]*(sc["mkt_scn"]/mkt_norm))
        units_pred = np.maximum(units_hat, 0).round()
        revenue_scn = float((sc["price_scn"] * units_pred).sum())
        profit_scn  = float((sc["price_scn"] * units_pred - (sc["unit_cost_scn"]*units_pred + 0.12*sc["mkt_scn"])).sum())
        return revenue_scn, profit_scn

    base_rev_s, base_profit_s = run_sum()
    tests = [
        ("قیمت +10%", dict(pm=price_mult*1.10)),
        ("قیمت -10%", dict(pm=price_mult*0.90)),
        ("مارکتینگ +20%", dict(mm=mkt_mult*1.20)),
        ("مارکتینگ -20%", dict(mm=mkt_mult*0.80)),
        ("هزینه واحد +10%", dict(cm=cost_mult*1.10)),
        ("هزینه واحد -10%", dict(cm=cost_mult*0.90)),
        ("کشش +0.1", dict(et=elastic_tweak+0.1)),
        ("کشش -0.1", dict(et=elastic_tweak-0.1)),
        ("حساسیت مارکتینگ +0.05", dict(bt=beta_tweak+0.05)),
        ("حساسیت مارکتینگ -0.05", dict(bt=beta_tweak-0.05)),
    ]
    sens = []
    for name, kwargs in tests:
        _, p = run_sum(**kwargs)
        sens.append({"پارامتر": name, "ΔProfit": p - base_profit_s})

    tornado = pd.DataFrame(sens).sort_values("ΔProfit", key=lambda s: s.abs(), ascending=False)
    st.plotly_chart(px.bar(tornado, y="پارامتر", x="ΔProfit", orientation="h", title="تورنادو حساسیت (اثر روی سود)"), use_container_width=True)

st.caption("نکته: می‌توانی این کد را با داده‌های واقعی جایگزین کنی و ضرایب را از مدل خودت بگیری.")
