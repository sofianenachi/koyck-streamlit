"""
تطبيق Streamlit لتوضيح فكرة التكامل المشترك (Cointegration)
- اختبار Engle-Granger
- اختبار Johansen
- محاكاة (بيانات مولَّدة أو بيانات المستخدم) + محاكاة مونتي كارلو

التشغيل:  streamlit run cointegration_app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.api import VAR

st.set_page_config(page_title="التكامل المشترك: Engle-Granger و Johansen", layout="wide")

st.markdown(
    """
<style>
h1, h2, h3, h4, .stMarkdown p, .stMarkdown li, .stAlert { direction: rtl; text-align: right; }
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# المحاكاة: نظام VECM
#   Δy_t = Π y_{t-1} + ε_t ,  Π = α β'   (رتبة Π = r = عدد علاقات التكامل المشترك)
# ----------------------------------------------------------------------------
def simulate_vecm(n, k, r, speed, beta_val, err_corr, seed, burn=100):
    rng = np.random.default_rng(seed)
    alpha = np.zeros((k, r))
    beta = np.zeros((k, r))
    if r > 0:
        beta[:r, :r] = np.eye(r)
        B = beta_val + 0.3 * rng.standard_normal((k - r, r))
        beta[r:, :] = -B
        alpha[:r, :r] = -speed * np.eye(r)
    pi = alpha @ beta.T
    cov = np.full((k, k), err_corr) + np.eye(k) * (1 - err_corr)
    chol = np.linalg.cholesky(cov)
    eps = rng.standard_normal((n + burn, k)) @ chol.T
    y = np.zeros((n + burn, k))
    for t in range(1, n + burn):
        y[t] = y[t - 1] + pi @ y[t - 1] + eps[t]
    return y[burn:], alpha, beta


def simulate_independent_ar(n, k, phi, seed):
    """سلاسل مستقرة I(0) مستقلة (للمقارنة)."""
    rng = np.random.default_rng(seed)
    y = np.zeros((n + 100, k))
    e = rng.standard_normal((n + 100, k))
    for t in range(1, n + 100):
        y[t] = phi * y[t - 1] + e[t]
    return y[100:]


def to_df(arr, names=None):
    names = names or [f"Y{i + 1}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=names)


# ----------------------------------------------------------------------------
# دوال الاختبارات
# ----------------------------------------------------------------------------
def adf_table(df, regression="c"):
    rows = []
    for c in df.columns:
        for label, s in [("المستوى", df[c]), ("الفرق الأول", df[c].diff().dropna())]:
            stat, p, _, _, crit, _ = adfuller(s, regression=regression, autolag="AIC")
            rows.append(
                {
                    "المتغير": c,
                    "الشكل": label,
                    "إحصائية ADF": round(stat, 3),
                    "القيمة الحرجة 5%": round(crit["5%"], 3),
                    "p-value": round(p, 4),
                    "القرار (5%)": "مستقرة I(0)" if p < 0.05 else "غير مستقرة (جذر وحدة)",
                }
            )
    return pd.DataFrame(rows)


def engle_granger(df, dep, trend="c"):
    regs = [c for c in df.columns if c != dep]
    X = sm.add_constant(df[regs]) if trend in ("c", "ct") else df[regs]
    ols = sm.OLS(df[dep], X).fit()
    t_stat, p_val, crit = coint(df[dep], df[regs], trend=trend, autolag="aic")
    resid = ols.resid
    # نموذج تصحيح الخطأ (ECM)
    d = df.diff().dropna()
    ecm_X = pd.concat([resid.shift(1).rename("ECT_{t-1}"), d[regs].add_prefix("Δ")], axis=1).dropna()
    ecm_y = d[dep].loc[ecm_X.index]
    ecm = sm.OLS(ecm_y, sm.add_constant(ecm_X)).fit()
    return ols, resid, t_stat, p_val, crit, ecm


def johansen(df, det_order, k_ar_diff):
    cj = coint_johansen(df, det_order, k_ar_diff)
    k = df.shape[1]
    trace = pd.DataFrame(
        {
            "H0: rank ≤": range(k),
            "Trace stat": cj.lr1.round(3),
            "CV 90%": cj.cvt[:, 0].round(3),
            "CV 95%": cj.cvt[:, 1].round(3),
            "CV 99%": cj.cvt[:, 2].round(3),
        }
    )
    trace["القرار (5%)"] = np.where(cj.lr1 > cj.cvt[:, 1], "رفض H0", "عدم رفض H0")
    maxeig = pd.DataFrame(
        {
            "H0: rank =": range(k),
            "Max-Eigen stat": cj.lr2.round(3),
            "CV 90%": cj.cvm[:, 0].round(3),
            "CV 95%": cj.cvm[:, 1].round(3),
            "CV 99%": cj.cvm[:, 2].round(3),
        }
    )
    maxeig["القرار (5%)"] = np.where(cj.lr2 > cj.cvm[:, 1], "رفض H0", "عدم رفض H0")
    rank = 0
    for i in range(k):
        if cj.lr1[i] > cj.cvt[i, 1]:
            rank += 1
        else:
            break
    return cj, trace, maxeig, rank


def monte_carlo(reps, n, k, r, speed, beta_val, err_corr, det_order, k_ar_diff, seed0):
    rows = []
    bar = st.progress(0.0)
    for i in range(reps):
        y, _, _ = simulate_vecm(n, k, r, speed, beta_val, err_corr, seed0 + i)
        p_eg = coint(y[:, 0], y[:, 1:], trend="c", autolag="aic")[1]
        ols = sm.OLS(y[:, 0], sm.add_constant(y[:, 1:])).fit()
        p_naive = ols.pvalues[1]
        cj = coint_johansen(y, det_order, k_ar_diff)
        rows.append(
            {
                "p_EG": p_eg,
                "p_naive_t": p_naive,
                "R2": ols.rsquared,
                "johansen_rank0_rejected": cj.lr1[0] > cj.cvt[0, 1],
            }
        )
        if i % 10 == 0 or i == reps - 1:
            bar.progress((i + 1) / reps)
    bar.empty()
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# الواجهة
# ----------------------------------------------------------------------------
st.title("التكامل المشترك: اختبارا Engle-Granger و Johansen مع محاكاة")
st.caption(
    "تطبيق تعليمي: ولِّد سلاسل زمنية بعلاقة تكامل مشترك معروفة، ثم اختبر قدرة الاختبارات على اكتشافها."
)

with st.sidebar:
    st.header("مصدر البيانات")
    source = st.radio("اختر", ["محاكاة", "رفع ملف CSV"])

    if source == "محاكاة":
        scenario = st.selectbox(
            "السيناريو",
            [
                "تكامل مشترك (VECM)",
                "انحدار زائف (سلاسل I(1) مستقلة)",
                "سلاسل مستقرة I(0) مستقلة",
            ],
        )
        n = st.slider("حجم العينة T", 50, 1000, 250, 10)
        k = st.slider("عدد المتغيرات k", 2, 4, 2)
        seed = st.number_input("البذرة العشوائية (seed)", value=42, step=1)
        if scenario == "تكامل مشترك (VECM)":
            if k > 2:
                r_true = st.slider("رتبة التكامل المشترك الحقيقية r", 1, k - 1, 1)
            else:
                r_true = 1
            speed = st.slider("سرعة التعديل |α| (نحو التوازن)", 0.01, 1.0, 0.3, 0.01)
            beta_val = st.slider("معامل العلاقة التوازنية β", 0.1, 3.0, 1.0, 0.1)
            err_corr = st.slider("ارتباط الصدمات بين المتغيرات", 0.0, 0.9, 0.0, 0.05)
        elif scenario.startswith("انحدار"):
            r_true, speed, beta_val, err_corr = 0, 0.0, 1.0, 0.0
        else:
            phi = st.slider("معامل الانحدار الذاتي φ", 0.0, 0.95, 0.5, 0.05)
    else:
        up = st.file_uploader("ملف CSV (أعمدة رقمية)", type=["csv"])

    st.header("إعدادات الاختبارات")
    eg_trend = st.selectbox(
        "حدود الاتجاه في Engle-Granger", ["c", "ct", "n"],
        format_func=lambda x: {"c": "ثابت", "ct": "ثابت واتجاه", "n": "بدون"}[x],
    )
    det_order = st.selectbox(
        "الحدود القطعية في Johansen", [-1, 0, 1],
        index=1,
        format_func=lambda x: {-1: "بدون (-1)", 0: "ثابت (0)", 1: "ثابت واتجاه (1)"}[x],
    )
    k_ar_diff = st.number_input("عدد فروق الإبطاء في Johansen (p-1)", 0, 12, 1)

# ----------------------------------------------------------------------------
# تحضير البيانات
# ----------------------------------------------------------------------------
true_info = None
if source == "محاكاة":
    if scenario == "تكامل مشترك (VECM)":
        arr, alpha_t, beta_t = simulate_vecm(n, k, r_true, speed, beta_val, err_corr, int(seed))
        true_info = (alpha_t, beta_t)
    elif scenario.startswith("انحدار"):
        arr, _, _ = simulate_vecm(n, k, 0, 0.0, 1.0, 0.0, int(seed))
        r_true = 0
    else:
        arr = simulate_independent_ar(n, k, phi, int(seed))
        r_true = None
    df = to_df(arr)
else:
    if up is None:
        st.info("ارفع ملف CSV من الشريط الجانبي، أو انتقل إلى وضع المحاكاة.")
        st.stop()
    raw = pd.read_csv(up)
    num_cols = raw.select_dtypes("number").columns.tolist()
    chosen = st.multiselect("اختر المتغيرات (2 على الأقل)", num_cols, default=num_cols[:2])
    if len(chosen) < 2:
        st.warning("اختر متغيرين على الأقل.")
        st.stop()
    df = raw[chosen].dropna().reset_index(drop=True)
    if st.checkbox("تحويل لوغاريتمي"):
        df = np.log(df)
    r_true = None
    k = df.shape[1]

dep = st.selectbox("المتغير التابع في Engle-Granger", df.columns.tolist(), index=0)

tab_data, tab_unit, tab_eg, tab_joh, tab_mc, tab_theory = st.tabs(
    ["البيانات", "جذر الوحدة (ADF)", "Engle-Granger", "Johansen", "مونتي كارلو", "شرح نظري"]
)

# --- البيانات
with tab_data:
    st.subheader("السلاسل الزمنية")
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for c in df.columns:
        ax.plot(df[c], label=c)
    ax.legend()
    ax.set_xlabel("t")
    st.pyplot(fig)
    if true_info is not None:
        alpha_t, beta_t = true_info
        st.markdown("**المعاملات الحقيقية المستخدمة في المحاكاة** (β هو متجه التكامل المشترك):")
        c1, c2 = st.columns(2)
        c1.write("α (سرعة التعديل)")
        c1.dataframe(pd.DataFrame(alpha_t).round(3))
        c2.write("β (علاقات التوازن)")
        c2.dataframe(pd.DataFrame(beta_t).round(3))
    elif source == "محاكاة" and r_true == 0:
        st.info("لا توجد علاقة تكامل مشترك حقيقية: كل سلسلة سير عشوائي مستقل.")
    with st.expander("عرض البيانات"):
        st.dataframe(df)
    st.download_button("تحميل البيانات CSV", df.to_csv(index=False).encode(), "data.csv")

# --- جذر الوحدة
with tab_unit:
    st.subheader("الخطوة الأولى: درجة تكامل كل سلسلة")
    st.write("شرط Engle-Granger: أن تكون السلاسل متكاملة من الرتبة نفسها، عادةً I(1).")
    reg = st.selectbox("نموذج ADF", ["c", "ct", "n"], format_func=lambda x: {"c": "ثابت", "ct": "ثابت واتجاه", "n": "بدون"}[x])
    st.dataframe(adf_table(df, reg), width='stretch')

# --- Engle-Granger
with tab_eg:
    st.subheader("اختبار Engle-Granger ذو الخطوتين")
    try:
        ols, resid, t_stat, p_val, crit, ecm = engle_granger(df, dep, eg_trend)
        st.markdown("**الخطوة 1:** انحدار العلاقة طويلة الأجل بالمربعات الصغرى")
        coefs = pd.DataFrame(
            {"المعامل": ols.params, "الخطأ المعياري": ols.bse, "t": ols.tvalues}
        ).round(4)
        st.dataframe(coefs)
        st.write(f"R² = {ols.rsquared:.4f}")
        if ols.rsquared > 0.7 and p_val > 0.05:
            st.warning("R² مرتفع مع عدم رفض عدم التكامل المشترك: علامة كلاسيكية على الانحدار الزائف.")

        st.markdown("**الخطوة 2:** اختبار جذر الوحدة على البواقي (قيم MacKinnon الحرجة)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("إحصائية EG", f"{t_stat:.3f}")
        c2.metric("CV 1%", f"{crit[0]:.3f}")
        c3.metric("CV 5%", f"{crit[1]:.3f}")
        c4.metric("CV 10%", f"{crit[2]:.3f}")
        st.write(f"p-value (MacKinnon) = **{p_val:.4f}**")
        if p_val < 0.05:
            st.success("نرفض H0 (عدم وجود تكامل مشترك) عند 5%: البواقي مستقرة، توجد علاقة توازنية طويلة الأجل.")
        else:
            st.error("لا نرفض H0 عند 5%: لا دليل على تكامل مشترك.")

        fig, axs = plt.subplots(1, 2, figsize=(11, 3.5))
        axs[0].plot(resid)
        axs[0].axhline(0, color="k", lw=0.8)
        axs[0].set_title("Residuals (equilibrium error)")
        pd.plotting.autocorrelation_plot(resid, ax=axs[1])
        axs[1].set_title("Residual ACF")
        st.pyplot(fig)

        st.markdown("**نموذج تصحيح الخطأ (ECM):** معامل حد تصحيح الخطأ يجب أن يكون سالبًا ومعنويًا إذا كان هناك تكامل مشترك.")
        ecm_tab = pd.DataFrame(
            {"المعامل": ecm.params, "t": ecm.tvalues, "p-value": ecm.pvalues}
        ).round(4)
        st.dataframe(ecm_tab)
        st.caption(
            "ملاحظة: اختبار t على معامل ECT لا يتبع التوزيع المعتاد تحت H0، ويُستخدم هنا للتوضيح فقط."
        )
        st.caption(
            "تنبيه: Engle-Granger يعتمد على اختيار المتغير التابع وقد يعطي نتائج مختلفة عند تبديله، ولا يكتشف أكثر من علاقة واحدة."
        )
    except Exception as e:
        st.error(f"تعذر تنفيذ الاختبار: {e}")

# --- Johansen
with tab_joh:
    st.subheader("اختبار Johansen (Trace و Max-Eigenvalue)")
    c1, c2 = st.columns([1, 3])
    if c1.button("اقتراح عدد الإبطاءات (AIC)"):
        try:
            sel = VAR(df.diff().dropna()).select_order(maxlags=8)
            p_aic = sel.selected_orders["aic"]
            st.info(f"AIC يقترح p = {p_aic} في VAR للفروق، أي k_ar_diff ≈ {max(p_aic - 0, 0)}.")
        except Exception as e:
            st.warning(f"تعذر اختيار الإبطاء: {e}")
    try:
        cj, trace, maxeig, rank = johansen(df, det_order, int(k_ar_diff))
        st.markdown("**اختبار الأثر (Trace)**")
        st.dataframe(trace, width='stretch')
        st.markdown("**اختبار القيمة الذاتية العظمى (Max-Eigenvalue)**")
        st.dataframe(maxeig, width='stretch')
        st.success(f"الرتبة المقدَّرة عند 5% (اختبار الأثر المتتابع): r = {rank}")
        if source == "محاكاة" and r_true is not None:
            st.write(f"الرتبة الحقيقية في المحاكاة: r = {r_true}")
        elif source == "محاكاة":
            st.write("الرتبة الحقيقية: السلاسل مستقرة (Π كاملة الرتبة)، وهذا يخالف افتراض I(1) في الاختبار.")

        if rank > 0:
            st.markdown("**متجهات التكامل المشترك (مُطبَّعة بحيث معامل المتغير الأول = 1)**")
            ev = cj.evec[:, :rank]
            ev = ev / ev[0, :]
            st.dataframe(
                pd.DataFrame(ev, index=df.columns, columns=[f"β{i + 1}" for i in range(rank)]).round(4)
            )
            ect = df.values @ ev
            fig, ax = plt.subplots(figsize=(10, 3))
            for i in range(rank):
                ax.plot(ect[:, i], label=f"β{i + 1}'y_t")
            ax.axhline(ect.mean(), color="k", lw=0.8)
            ax.legend()
            ax.set_title("Estimated cointegrating relations (should look stationary)")
            st.pyplot(fig)
        st.caption(
            "إذا اختلف اختبارا Trace و Max-Eigen فالمعتاد الاعتماد على Trace مع الحذر، وتتأثر النتائج بالإبطاء والحدود القطعية."
        )
    except Exception as e:
        st.error(f"تعذر تنفيذ اختبار Johansen: {e}")

# --- مونتي كارلو
with tab_mc:
    st.subheader("محاكاة مونتي كارلو: حجم الاختبار وقوته")
    if source != "محاكاة" or r_true is None:
        st.info("هذه التجربة متاحة في سيناريوهي التكامل المشترك والانحدار الزائف من وضع المحاكاة.")
    else:
        reps = st.slider("عدد التكرارات", 50, 1000, 200, 50)
        if st.button("تشغيل المحاكاة"):
            if scenario == "تكامل مشترك (VECM)":
                p_args = (n, k, r_true, speed, beta_val, err_corr)
            else:
                p_args = (n, k, 0, 0.0, 1.0, 0.0)
            with st.spinner("جارٍ التشغيل..."):
                h0 = monte_carlo(reps, n, k, 0, 0.0, 1.0, 0.0, det_order, int(k_ar_diff), 1000)
                h1 = monte_carlo(reps, *p_args, det_order, int(k_ar_diff), 5000) if r_true > 0 else None
            st.markdown("**تحت H0: لا تكامل مشترك (سير عشوائي مستقل)**")
            res = {
                "Engle-Granger (رفض عند 5%)": (h0.p_EG < 0.05).mean(),
                "Johansen Trace r=0 (رفض عند 5%)": h0.johansen_rank0_rejected.mean(),
                "t-test الساذج (انحدار زائف)": (h0.p_naive_t < 0.05).mean(),
            }
            st.dataframe(pd.DataFrame({"نسبة الرفض": res}).round(3))
            st.caption("المطلوب نظريًا ≈ 5%، أما اختبار t الساذج فيرفض أكثر بكثير: هذه مشكلة الانحدار الزائف.")
            if h1 is not None:
                st.markdown("**تحت H1: يوجد تكامل مشترك (القوة الإحصائية)**")
                res1 = {
                    "Engle-Granger": (h1.p_EG < 0.05).mean(),
                    "Johansen Trace r=0": h1.johansen_rank0_rejected.mean(),
                }
                st.dataframe(pd.DataFrame({"القوة (نسبة الرفض)": res1}).round(3))
            fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))
            axs[0].hist(h0.p_EG, bins=20, alpha=0.7, label="H0 (no coint.)")
            if h1 is not None:
                axs[0].hist(h1.p_EG, bins=20, alpha=0.7, label="H1 (coint.)")
            axs[0].set_title("Engle-Granger p-values")
            axs[0].legend()
            axs[1].hist(h0.R2, bins=20, color="tab:red", alpha=0.7)
            axs[1].set_title("R² in spurious regressions")
            st.pyplot(fig)

# --- شرح نظري
with tab_theory:
    st.markdown(
        r"""
### فكرة التكامل المشترك
سلسلتان $y_t, x_t$ كلتاهما $I(1)$ (غير مستقرتين)، لكن توجد توليفة خطية $u_t = y_t - \beta x_t$ مستقرة $I(0)$.
أي أن السلسلتين قد تتباعدان مؤقتًا لكنهما تتحركان معًا في المدى الطويل.

### Engle-Granger
1. اختبار أن السلاسل $I(1)$.
2. تقدير $y_t = c + \beta x_t + u_t$ بالمربعات الصغرى.
3. اختبار جذر الوحدة على $\hat u_t$ بقيم MacKinnon الحرجة (وليست قيم Dickey-Fuller العادية).
- $H_0$: لا تكامل مشترك. رفض $H_0$ يعني وجود علاقة توازنية.
- نقاط الضعف: علاقة واحدة فقط، ويتأثر بالمتغير التابع المختار.

### Johansen
يقدّر نظام VECM: $\Delta y_t = \Pi y_{t-1} + \sum_i \Gamma_i \Delta y_{t-i} + \varepsilon_t$ ، حيث $\Pi = \alpha\beta'$ ورتبتها $r$.
- اختبار الأثر: $H_0$ أن عدد العلاقات $\le r$.
- اختبار القيمة الذاتية العظمى: $H_0$ أن العدد $= r$ مقابل $r+1$.
- يكتشف عدة علاقات ويعامل المتغيرات بتماثل، لكنه حساس للإبطاء وللعينات الصغيرة.

### الانحدار الزائف
انحدار سلسلتين $I(1)$ مستقلتين يعطي غالبًا $R^2$ مرتفعًا وإحصائية t معنوية ظاهريًا (Granger وNewbold، 1974).
تبيّن تجربة مونتي كارلو ذلك عمليًا.

### نموذج تصحيح الخطأ
$\Delta y_t = c + \gamma\,\hat u_{t-1} + \delta\,\Delta x_t + e_t$ ، ويدل $\gamma<0$ على أن الانحراف عن التوازن يُصحَّح تدريجيًا.
"""
    )
