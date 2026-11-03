import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from datetime import date

st.set_page_config(page_title="المحلل الذكي للبورصة المصرية", page_icon="📈", layout="wide")
st.title("📈 المحلل الذكي للبورصة المصرية")
st.caption("نسخة أولية قابلة للتشغيل على الويب — تحليل تاريخي وتعليمي، وليست توصية استثمارية.")

with st.sidebar:
    st.header("إعدادات التحليل")
    symbols_text = st.text_area("رموز الأسهم (رمز في كل سطر)", "COMI.CA\nSWDY.CA\nETEL.CA\nTMGH.CA\nABUK.CA")
    period = st.selectbox("الفترة التاريخية", ["2y", "5y", "10y"], index=1)
    test_fraction = st.slider("نسبة بيانات الاختبار", 0.2, 0.4, 0.3, 0.05)
    cost_pct = st.number_input("تكلفة التداول التقديرية لكل تغيير مركز (%)", min_value=0.0, max_value=2.0, value=0.2, step=0.05)
    run = st.button("🔄 جلب البيانات وتحليل الأسهم", type="primary", use_container_width=True)

st.warning("البيانات عبر Yahoo Finance قد تكون متأخرة أو ناقصة، وتوفر رموز الأسهم المصرية ليس مضمونًا. لا تعتبر النتائج أسعارًا لحظية أو توصيات شراء. تحقق من السعر ومصدره لدى وسيطك قبل أي قرار.")

FEATURES = ["ret1", "ret5", "ret10", "vol10", "vol20", "ma_gap10", "ma_gap20", "range5"]
def features_from_close(close):
    df = pd.DataFrame(index=close.index)
    df["ret1"] = close.pct_change(1)
    df["ret5"] = close.pct_change(5)
    df["ret10"] = close.pct_change(10)
    df["vol10"] = df["ret1"].rolling(10).std()
    df["vol20"] = df["ret1"].rolling(20).std()
    df["ma_gap10"] = close / close.rolling(10).mean() - 1
    df["ma_gap20"] = close / close.rolling(20).mean() - 1
    df["range5"] = close.pct_change().rolling(5).max() - close.pct_change().rolling(5).min()
    # Label is whether next session's close is higher than today's close
    df["target"] = (close.shift(-1) > close).astype(int)
    return df.replace([np.inf, -np.inf], np.nan).dropna()

def max_drawdown(equity):
    if len(equity) == 0: return np.nan
    peak = equity.cummax()
    return float((equity / peak - 1).min() * 100)

def analyze(symbol, period, test_fraction, cost_pct):
    raw = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False, threads=False)
    if raw is None or raw.empty:
        return None, f"لم تُرجع خدمة البيانات أسعارًا للرمز {symbol}."
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    if "Close" not in raw.columns:
        return None, f"عمود الإغلاق غير متاح للرمز {symbol}."
    close = raw["Close"].dropna()
    data = features_from_close(close)
    if len(data) < 120:
        return None, f"البيانات الصالحة قليلة للرمز {symbol} ({len(data)} جلسة فقط)."
    X = data[FEATURES]
    y = data["target"]
    split = int(len(data) * (1-test_fraction))
    if split < 80 or len(data)-split < 20:
        return None, f"لا تكفي البيانات لتقسيم تدريب/اختبار للرمز {symbol}."
    model = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=5,
                                   class_weight="balanced_subsample", random_state=42)
    model.fit(X.iloc[:split], y.iloc[:split])
    X_test, y_test = X.iloc[split:], y.iloc[split:]
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    classes = list(model.classes_)
    up_idx = classes.index(1) if 1 in classes else None
    p_up = float(model.predict_proba(X.iloc[[-1]])[0][up_idx]) if up_idx is not None else 0.0
    acc = accuracy_score(y_test, pred) * 100
    bacc = balanced_accuracy_score(y_test, pred) * 100
    # Prediction formed after close at t; position is applied to return from t to t+1.
    # Test returns are next-session returns for each feature row, computed from close[t+1]/close[t]-1.
    test_index = X_test.index
    close_aligned = close.reindex(data.index)
    next_returns = close_aligned.shift(-1).div(close_aligned).sub(1).reindex(test_index)
    positions = pd.Series(pred, index=test_index).astype(float)
    turnover = positions.diff().abs()
    if len(turnover): turnover.iloc[0] = abs(positions.iloc[0])
    net_strategy = positions * next_returns - (turnover * cost_pct/100)
    net_strategy = net_strategy.dropna()
    bh_returns = next_returns.loc[net_strategy.index].fillna(0)
    strat_equity = (1 + net_strategy).cumprod()
    bh_equity = (1 + bh_returns).cumprod()
    total_strat = (strat_equity.iloc[-1]-1)*100 if len(strat_equity) else np.nan
    total_bh = (bh_equity.iloc[-1]-1)*100 if len(bh_equity) else np.nan
    dd_strat = max_drawdown(strat_equity)
    dd_bh = max_drawdown(bh_equity)
    ann_vol = float(data.loc[test_index, "ret1"].std() * np.sqrt(252) * 100)
    row = {
        "السهم": symbol.replace(".CA",""),
        "آخر إغلاق متاح": float(close.iloc[-1]),
        "تاريخ آخر بيانات": str(close.index[-1].date()),
        "احتمال صعود تقديري للجلسة التالية %": round(p_up*100,2),
        "دقة الاختبار %": round(acc,2),
        "الدقة المتوازنة %": round(bacc,2),
        "عائد النموذج بعد التكلفة %": round(total_strat,2),
        "أكبر تراجع للنموذج %": round(dd_strat,2),
        "عائد الاحتفاظ خلال الاختبار %": round(total_bh,2),
        "أكبر تراجع للاحتفاظ %": round(dd_bh,2),
        "التقلب السنوي التقديري %": round(ann_vol,2),
        "جلسات الاختبار": len(net_strategy)
    }
    # Add only a cautious interpretation, never a buy recommendation.
    if p_up >= .60 and bacc >= 52 and total_strat > total_bh and dd_strat > -30:
        assessment = "إشارة تستحق مراجعة إضافية"
    elif bacc < 50 or total_strat <= total_bh:
        assessment = "نتائج غير حاسمة / تحتاج تحسين"
    else:
        assessment = "مراقبة فقط"
    row["قراءة مبدئية"] = assessment
    chart = pd.DataFrame({"استراتيجية النموذج": strat_equity, "الاحتفاظ": bh_equity})
    return (row, chart), None

if run:
    symbols = list(dict.fromkeys([s.strip().upper() for s in symbols_text.splitlines() if s.strip()]))
    if not symbols:
        st.error("اكتبي رمز سهم واحدًا على الأقل.")
    else:
        results, errors, charts = [], [], {}
        bar = st.progress(0, text="بدء التحليل...")
        for i, sym in enumerate(symbols):
            bar.progress((i+1)/len(symbols), text=f"تحليل {sym}...")
            try:
                out, err = analyze(sym, period, test_fraction, cost_pct)
                if err: errors.append(err)
                else:
                    row, chart = out
                    results.append(row)
                    charts[row["السهم"]] = chart
            except Exception as e:
                errors.append(f"تعذر تحليل {sym}: {str(e)[:180]}")
        bar.empty()
        if results:
            df = pd.DataFrame(results).sort_values("احتمال صعود تقديري للجلسة التالية %", ascending=False)
            st.subheader("نتائج التحليل")
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("⬇️ تنزيل النتائج CSV", df.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"egx_ai_report_{date.today().isoformat()}.csv", mime="text/csv")
            st.subheader("مقارنة منحنى الاختبار")
            selected = st.selectbox("اختاري سهمًا لعرض المنحنى", list(charts.keys()))
            st.line_chart(charts[selected], use_container_width=True)
            st.info("الاحتمال يخص الجلسة التالية وفق النموذج، وليس ضمانًا. الاختبار التاريخي لا يضمن الأداء المستقبلي، كما أن جودة البيانات وتكاليف التنفيذ قد تختلف عن الافتراضات.")
        if errors:
            st.subheader("ملاحظات على البيانات")
            for msg in errors: st.warning(msg)
else:
    st.info("اضغطي «جلب البيانات وتحليل الأسهم» لبدء التحليل. يتطلب الاتصال بالإنترنت ومصدر البيانات أن يكون متاحًا.")
    st.subheader("ما الذي يفعله النموذج؟")
    st.markdown("""
    - يبني ميزات من عوائد الأسعار والتقلب والمتوسطات المتحركة.
    - يتدرب على الجزء الأقدم من البيانات ويختبر على جزء أحدث.
    - يقارن استراتيجية طويلة فقط بالاحتفاظ بالسهم، مع تكلفة تداول افتراضية.
    - يعرض تاريخ آخر بيانات لتقليل خطر الاعتماد على أسعار قديمة.
    """)
st.caption("نسخة تجريبية: لا تنفذ أوامر بيع أو شراء، ولا تطلب بيانات دخول للوسيط، ولا تحفظ مفاتيح أو معلومات مالية شخصية.")
