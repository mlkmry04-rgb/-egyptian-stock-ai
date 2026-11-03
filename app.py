import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date

st.set_page_config(page_title="المحلل الذكي للبورصة المصرية", page_icon="📈", layout="wide")

st.title("📈 المحلل الذكي للبورصة المصرية")
st.caption("أداة تحليل ومتابعة تعليمية — ليست توصية استثمارية ولا تنفذ أوامر تداول.")

DEFAULTS = ["COMI.CA", "SWDY.CA", "ETEL.CA", "TMGH.CA", "ABUK.CA"]

@st.cache_data(ttl=900, show_spinner=False)
def load_data(symbol: str, period: str):
    df = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in needed):
        return pd.DataFrame()
    df = df[needed].dropna(subset=["Close"]).copy()
    return df

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def analyze(df):
    x = df.copy()
    c = x["Close"]
    x["SMA20"] = c.rolling(20).mean()
    x["SMA50"] = c.rolling(50).mean()
    x["RSI"] = rsi(c)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    x["MACD"] = ema12 - ema26
    x["MACD_Signal"] = x["MACD"].ewm(span=9, adjust=False).mean()
    tr = pd.concat([
        x["High"] - x["Low"],
        (x["High"] - c.shift()).abs(),
        (x["Low"] - c.shift()).abs()
    ], axis=1).max(axis=1)
    x["ATR"] = tr.rolling(14).mean()
    return x.dropna().copy()

def signal_for(row):
    score = 0
    reasons = []
    if row["Close"] > row["SMA20"]:
        score += 1; reasons.append("السعر أعلى متوسط 20 يوم")
    else:
        score -= 1; reasons.append("السعر أقل متوسط 20 يوم")
    if row["SMA20"] > row["SMA50"]:
        score += 1; reasons.append("متوسط 20 أعلى متوسط 50")
    else:
        score -= 1; reasons.append("متوسط 20 أقل متوسط 50")
    if row["MACD"] > row["MACD_Signal"]:
        score += 1; reasons.append("تقاطع MACD إيجابي")
    else:
        score -= 1; reasons.append("MACD دون خط الإشارة")
    if row["RSI"] < 30:
        score += 1; reasons.append("RSI يشير إلى تشبع بيعي")
    elif row["RSI"] > 70:
        score -= 1; reasons.append("RSI يشير إلى تشبع شرائي")
    if score >= 2:
        label = "إيجابية بحذر"
    elif score <= -2:
        label = "سلبية / تجنب التسرع"
    else:
        label = "انتظار"
    return label, score, reasons

st.sidebar.header("إعدادات التحليل")
symbols_text = st.sidebar.text_area("رموز الأسهم (افصلي بينها بفاصلة)", ", ".join(DEFAULTS))
symbols = list(dict.fromkeys(s.strip().upper() for s in symbols_text.replace("\n", ",").split(",") if s.strip()))
period = st.sidebar.selectbox("الفترة التاريخية", ["6mo", "1y", "2y", "5y"], index=3)
cost_pct = st.sidebar.number_input("تكلفة التداول التقديرية لكل عملية (%)", min_value=0.0, max_value=5.0, value=0.20, step=0.05)
st.sidebar.caption("مصدر الأسعار: Yahoo Finance عبر yfinance. قد لا تتوفر بعض رموز EGX أو قد تتأخر البيانات.")

if st.button("🔄 جلب البيانات وتحليل الأسهم", type="primary", use_container_width=True):
    st.cache_data.clear()

if not symbols:
    st.info("أضيفي رمز سهم واحدًا على الأقل.")
    st.stop()

rows = []
details = []
progress = st.progress(0)
status = st.empty()

for i, sym in enumerate(symbols):
    status.write(f"جاري تحميل وتحليل {sym} ...")
    try:
        raw = load_data(sym, period)
        if raw.empty or len(raw) < 60:
            details.append((sym, None, "لا توجد بيانات كافية أو الرمز غير مدعوم من مصدر البيانات."))
        else:
            data = analyze(raw)
            if data.empty:
                details.append((sym, None, "البيانات غير كافية لحساب المؤشرات."))
            else:
                last = data.iloc[-1]
                label, score, reasons = signal_for(last)
                atr = float(last["ATR"])
                close = float(last["Close"])
                # مستويات تعليمية مبنية على ATR وليست توقعات مؤكدة
                stop = max(0.0, close - 1.5 * atr)
                target = close + 2.0 * atr
                rows.append({
                    "السهم": sym,
                    "آخر إغلاق": round(close, 3),
                    "تاريخ البيانات": data.index[-1].strftime("%Y-%m-%d"),
                    "RSI": round(float(last["RSI"]), 1),
                    "MACD": round(float(last["MACD"]), 3),
                    "متوسط 20": round(float(last["SMA20"]), 3),
                    "متوسط 50": round(float(last["SMA50"]), 3),
                    "الإشارة الفنية": label,
                    "درجة الإشارة": score,
                    "وقف خسارة تعليمي": round(stop, 3),
                    "هدف تعليمي": round(target, 3),
                })
                details.append((sym, data, reasons))
    except Exception as e:
        details.append((sym, None, f"تعذر التحميل: {str(e)[:160]}"))
    progress.progress((i + 1) / len(symbols))

progress.empty()
status.empty()

if rows:
    result = pd.DataFrame(rows).sort_values("درجة الإشارة", ascending=False)
    st.subheader("📊 ملخص الأسهم")
    st.dataframe(result, use_container_width=True, hide_index=True)
    st.download_button("⬇️ تنزيل النتائج CSV", result.to_csv(index=False).encode("utf-8-sig"),
                       "egx_stock_analysis.csv", "text/csv")

    st.subheader("🔎 تفاصيل كل سهم")
    for sym, data, extra in details:
        with st.expander(f"{sym} — التفاصيل", expanded=False):
            if data is None:
                st.warning(extra)
                continue
            last = data.iloc[-1]
            label, score, reasons = signal_for(last)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("آخر إغلاق", f'{last["Close"]:.3f}')
            c2.metric("RSI", f'{last["RSI"]:.1f}')
            c3.metric("MACD", f'{last["MACD"]:.3f}')
            c4.metric("الإشارة", label)
            st.write("**أسباب الإشارة:**")
            for reason in reasons:
                st.write(f"- {reason}")
            st.line_chart(data[["Close", "SMA20", "SMA50"]], use_container_width=True)
            st.caption(f"آخر تاريخ متاح: {data.index[-1].strftime('%Y-%m-%d')} | ATR: {last['ATR']:.3f}")
            st.caption("مستويات ATR إرشادية فقط؛ لا تأخذ في الاعتبار فجوات الأسعار أو حدود التذبذب أو السيولة.")

elif details:
    st.subheader("نتيجة جلب البيانات")
    for sym, _, msg in details:
        st.warning(f"{sym}: {msg}")

st.divider()
st.warning(
    "تنبيه مهم: الإشارات والمؤشرات مبنية على بيانات تاريخية وقد تكون متأخرة أو غير مكتملة. "
    "لا تمثل توصية شراء أو بيع، ولا تضمن الربح. تحققي من الأسعار والسيولة ومواعيد التداول "
    "ومصدر بيانات موثوق قبل اتخاذ أي قرار، ولا تستثمري أموالًا لا تتحملين خسارتها."
)
