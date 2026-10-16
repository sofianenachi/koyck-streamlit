import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf
from arch.unitroot import PhillipsPerron

warnings.filterwarnings("ignore")
st.set_page_config(page_title="استقرارية السلاسل الزمنية", page_icon="📈", layout="wide")
st.markdown(
    "<style>html, body, [class*='st-'] {direction: rtl; text-align: right;}"
    ".stMarkdown .katex-display {direction: ltr;}</style>",
    unsafe_allow_html=True,
)

KINDS = {
    "white": "ضجيج أبيض (White Noise)",
    "ar1": "AR(1) مستقرة",
    "ts": "اتجاه حتمي + ضجيج (TS)",
    "rw": "سير عشوائي (Random Walk)",
    "rwd": "سير عشوائي مع انحراف (Drift)",
    "hetero": "تباين متغير عبر الزمن",
    "seasonal": "موسمية + ضجيج",
}


def simulate(kind, n=200, seed=1, phi=0.6, drift=0.2, slope=0.1, sigma=1.0):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, sigma, n)
    t = np.arange(n)
    if kind == "white":
        y = e
    elif kind == "ar1":
        y = np.zeros(n)
        for i in range(1, n):
            y[i] = phi * y[i - 1] + e[i]
    elif kind == "ts":
        y = slope * t + e
    elif kind == "rw":
        y = np.cumsum(e)
    elif kind == "rwd":
        y = np.cumsum(drift + e)
    elif kind == "hetero":
        y = e * np.linspace(1, 5, n)
    else:
        y = 3 * np.sin(2 * np.pi * t / 12) + e
    return pd.Series(y, name=kind)


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def plot_diag(y, w=20, title=""):
    """Series + rolling mean/std band + ACF."""
    y = pd.Series(y).reset_index(drop=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [2, 1]})
    m, s = y.rolling(w).mean(), y.rolling(w).std()
    ax[0].plot(y, lw=1, label="Series")
    ax[0].plot(m, c="red", lw=2, label=f"Rolling mean ({w})")
    ax[0].fill_between(y.index, m - 2 * s, m + 2 * s, color="red", alpha=0.12, label="±2 rolling std")
    ax[0].set_title(title)
    ax[0].legend(loc="upper left", fontsize=8)
    plot_acf(y.dropna(), ax=ax[1], lags=min(30, len(y) // 3), title="ACF")
    fig.tight_layout()
    show(fig)


def sidebar_sim(key, kinds=KINDS, default=0):
    kind = st.selectbox("نوع السلسلة", list(kinds), index=default, format_func=kinds.get, key=key + "k")
    c1, c2 = st.columns(2)
    n = c1.slider("عدد المشاهدات T", 50, 1000, 250, 10, key=key + "n")
    seed = c2.number_input("البذرة (seed)", 0, 9999, 1, key=key + "s")
    return kind, n, int(seed)


# ------------------------------------------------------------------ pages
def page_home():
    st.title("📈 استقرارية السلاسل الزمنية")
    st.subheader("محاضرة تفاعلية: المفاهيم، السيرورات العشوائية، واختبارات جذر الوحدة")
    st.markdown(
        """
**محاور المحاضرة:**
1. **مفهوم الاستقرارية** – لماذا تهمّنا؟ وكيف نكشفها بصرياً؟
2. **أنواع الاستقرارية** – الصارمة والضعيفة، و TS مقابل DS
3. **السيرورات العشوائية** – ضجيج أبيض، AR(1)، سير عشوائي
4. **اختبارات جذر الوحدة** – ADF و PP و KPSS (النظرية)
5. **المخبر التطبيقي** – طبّق الاختبارات على بيانات محاكاة أو بياناتك (CSV)

> استخدم القائمة الجانبية للتنقل، وحرّك المنزلقات لترى أثر كل معلمة مباشرة.
"""
    )
    st.info("ملاحظة: عناوين الرسوم بالإنجليزية لتفادي مشاكل عرض الحروف العربية في matplotlib.")


def page_concept():
    st.header("1) مفهوم الاستقرارية")
    st.markdown(
        "تكون السلسلة $y_t$ **مستقرة** إذا لم تتغير خصائصها الإحصائية عبر الزمن. "
        "عدم الاستقرارية يؤدي إلى **انحدار زائف (Spurious Regression)** ونتائج غير موثوقة."
    )
    kind, n, seed = sidebar_sim("c")
    w = st.slider("نافذة المتوسط المتحرك", 5, 60, 20)
    y = simulate(kind, n, seed)
    plot_diag(y, w, KINDS[kind].split(" (")[0] and kind)
    h = n // 2
    a, b = y[:h], y[h:]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("متوسط النصف الأول", f"{a.mean():.2f}")
    c2.metric("متوسط النصف الثاني", f"{b.mean():.2f}", f"{b.mean() - a.mean():.2f}")
    c3.metric("تباين النصف الأول", f"{a.var():.2f}")
    c4.metric("تباين النصف الثاني", f"{b.var():.2f}", f"{b.var() - a.var():.2f}")
    st.markdown(
        "**كيف نقرأ الرسم؟** إذا تحرك المتوسط المتحرك (الأحمر) أو اتسع الشريط، فهناك شك في الاستقرارية. "
        "وإذا تناقص الـ ACF ببطء شديد فهذا مؤشر على جذر وحدة."
    )
    with st.expander("الشروط الرياضية للاستقرارية الضعيفة"):
        st.latex(r"E(y_t)=\mu \;\; \forall t")
        st.latex(r"Var(y_t)=\sigma^2<\infty \;\; \forall t")
        st.latex(r"Cov(y_t,y_{t-k})=\gamma_k \;\; \text{(تتعلق بالفجوة } k \text{ فقط)}")


def page_types():
    st.header("2) أنواع الاستقرارية")
    st.markdown(
        """
- **الاستقرارية الصارمة (Strict):** التوزيع المشترك لـ $(y_{t_1},\\dots,y_{t_k})$ لا يتغير بإزاحة الزمن. شرط قوي نادر التحقق عملياً.
- **الاستقرارية الضعيفة (Weak / Covariance):** ثبات المتوسط والتباين والتغاير بين الفترات. وهي المستعملة عملياً.
- إذا كانت السلسلة غوسية فالضعيفة ⇔ الصارمة.
"""
    )
    st.subheader("TS مقابل DS: كيف نجعل السلسلة مستقرة؟")
    st.latex(r"TS:\; y_t=a+bt+\varepsilon_t \qquad DS:\; y_t=y_{t-1}+b+\varepsilon_t")
    c1, c2, c3 = st.columns(3)
    n = c1.slider("T", 100, 600, 250, 10, key="tn")
    b = c2.slider("الميل / الانحراف b", 0.0, 0.5, 0.15, 0.01)
    seed = c3.number_input("seed", 0, 9999, 3, key="ts_seed")
    t = np.arange(n)
    ts = simulate("ts", n, int(seed), slope=b)
    ds = simulate("rwd", n, int(seed), drift=b)
    fig, ax = plt.subplots(2, 3, figsize=(13, 6))
    for r, (name, y) in enumerate([("TS", ts), ("DS", ds)]):
        resid = y - np.polyval(np.polyfit(t, y, 1), t)
        ax[r, 0].plot(y.values); ax[r, 0].set_title(f"{name}: original")
        ax[r, 1].plot(resid.values, c="green"); ax[r, 1].set_title(f"{name}: after DETRENDING")
        ax[r, 2].plot(np.diff(y.values), c="purple"); ax[r, 2].set_title(f"{name}: after DIFFERENCING")
    fig.tight_layout()
    show(fig)
    st.success(
        "**الخلاصة:** سلسلة TS تُجعل مستقرة بـ **نزع الاتجاه** (والفرق الأول يضيف لها ارتباطاً زائفاً MA). "
        "سلسلة DS تُجعل مستقرة بـ **الفروق**، بينما نزع الاتجاه يترك الصدمات تتراكم (ما زال المتبقي غير مستقر). "
        "وهنا تحديداً تأتي أهمية اختبارات جذر الوحدة."
    )


def page_stochastic():
    st.header("3) السيرورات العشوائية (Stochastic Processes)")
    st.markdown(
        "السيرورة العشوائية هي **عائلة متغيرات عشوائية** $\\{y_t, t\\in T\\}$، والسلسلة المشاهدة "
        "ما هي إلا **تحقق واحد (Realization)** منها."
    )
    procs = {"white": KINDS["white"], "ar1": "AR(1)", "rw": KINDS["rw"], "rwd": KINDS["rwd"]}
    c1, c2, c3, c4 = st.columns(4)
    kind = c1.selectbox("السيرورة", list(procs), format_func=procs.get)
    paths = c2.slider("عدد التحققات", 1, 40, 12)
    phi = c3.slider("φ (لـ AR(1))", -1.1, 1.1, 0.7, 0.05)
    n = c4.slider("T", 50, 500, 200, 10, key="sn")
    drift = 0.1
    t = np.arange(1, n + 1)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for i in range(paths):
        y = simulate(kind, n, i, phi=phi, drift=drift)
        ax[0].plot(y.values, lw=0.8, alpha=0.7)
    if kind == "rw":
        sd, mean = np.sqrt(t), np.zeros(n)
    elif kind == "rwd":
        sd, mean = np.sqrt(t), drift * t
    elif kind == "ar1" and abs(phi) < 1:
        sd, mean = np.full(n, 1 / np.sqrt(1 - phi**2)), np.zeros(n)
    elif kind == "ar1":
        sd, mean = np.sqrt((np.abs(phi) ** (2 * t) - 1) / (phi**2 - 1)) if abs(phi) != 1 else np.sqrt(t), np.zeros(n)
    else:
        sd, mean = np.ones(n), np.zeros(n)
    ax[0].plot(mean + 2 * sd, "k--"); ax[0].plot(mean - 2 * sd, "k--")
    ax[0].set_title("Realizations with theoretical ±2 std band")
    ax[1].plot(sd**2, c="red"); ax[1].set_title("Theoretical Var(y_t)")
    fig.tight_layout()
    show(fig)
    if kind == "ar1":
        st.latex(r"y_t=\phi\, y_{t-1}+\varepsilon_t")
        if abs(phi) < 1:
            st.success(f"|φ|={abs(phi):.2f} < 1 ⇒ مستقرة؛ التباين ثابت = 1/(1-φ²).")
        elif abs(phi) == 1:
            st.warning("φ = 1 ⇒ جذر وحدة (سير عشوائي).")
        else:
            st.error("|φ| > 1 ⇒ سيرورة انفجارية.")
    elif kind in ("rw", "rwd"):
        st.latex(r"y_t=y_0+\sum_{i=1}^{t}\varepsilon_i \;\Rightarrow\; Var(y_t)=t\,\sigma^2")
        st.warning("التباين يزداد خطياً مع الزمن ⇒ غير مستقرة، وتأثير الصدمة دائم.")
    else:
        st.info("ضجيج أبيض: متوسط 0، تباين ثابت، بدون ارتباط ذاتي. هو لبنة بناء النماذج.")


def page_tests_theory():
    st.header("4) اختبارات جذر الوحدة")
    tabs = st.tabs(["ADF", "Phillips-Perron", "KPSS", "مقارنة"])
    with tabs[0]:
        st.markdown("**Augmented Dickey-Fuller:** نضيف فروقاً متباطئة لامتصاص الارتباط الذاتي في الأخطاء.")
        st.latex(r"\Delta y_t=\alpha+\beta t+\gamma y_{t-1}+\sum_{i=1}^{p}\delta_i\Delta y_{t-i}+\varepsilon_t")
        st.latex(r"H_0:\gamma=0 \;(\text{جذر وحدة}) \qquad H_1:\gamma<0 \;(\text{مستقرة})")
        st.markdown("نتائج الاختبار تُقارن بقيم **Dickey-Fuller الحرجة** (وليس Student). عدد الفجوات $p$ يُختار بمعيار AIC/BIC.")
    with tabs[1]:
        st.markdown(
            "**Phillips-Perron:** نفس معادلة DF بدون فروق متباطئة، لكن **يصحّح إحصائية t** بشكل لامعلمي "
            "(مقدّر Newey-West) لمعالجة الارتباط الذاتي وعدم ثبات التباين."
        )
        st.latex(r"y_t=\alpha+\beta t+\rho\, y_{t-1}+u_t \qquad H_0:\rho=1")
        st.markdown("نفس فرضيات وقيم ADF الحرجة، ويتفوق عندما يكون التباين غير ثابت.")
    with tabs[2]:
        st.markdown("**KPSS:** يعكس الفرضيات! الأصل هو الاستقرارية.")
        st.latex(r"y_t=\xi t+r_t+\varepsilon_t,\quad r_t=r_{t-1}+u_t,\; u_t\sim(0,\sigma_u^2)")
        st.latex(r"H_0:\sigma_u^2=0 \;(\text{مستقرة}) \qquad H_1:\sigma_u^2>0 \;(\text{جذر وحدة})")
        st.latex(r"LM=\frac{\sum_{t=1}^{T}S_t^2}{T^2\,\hat\sigma^2},\quad S_t=\sum_{i=1}^{t}\hat e_i")
    with tabs[3]:
        st.table(
            pd.DataFrame(
                {
                    "ADF": ["جذر وحدة", "γ < 0 (مستقرة)", "p < α", "معلمي (فروق متباطئة)"],
                    "PP": ["جذر وحدة", "ρ < 1 (مستقرة)", "p < α", "لامعلمي (تصحيح Newey-West)"],
                    "KPSS": ["مستقرة", "جذر وحدة", "p < α", "LM / لامعلمي"],
                },
                index=["H0", "H1", "نرفض H0 إذا", "معالجة الارتباط الذاتي"],
            )
        )
        st.markdown("#### جدول القرار المشترك")
        st.table(
            pd.DataFrame(
                {
                    "ADF / PP": ["رفض H0", "عدم رفض H0", "رفض H0", "عدم رفض H0"],
                    "KPSS": ["عدم رفض H0", "رفض H0", "رفض H0", "عدم رفض H0"],
                    "الاستنتاج": [
                        "مستقرة ✅",
                        "غير مستقرة (جذر وحدة) ❌",
                        "تعارض: غالباً نموذج غير ملائم أو استقرارية حول اتجاه",
                        "ضعف قدرة الاختبارات (عينة صغيرة)",
                    ],
                }
            )
        )


def run_tests(x, reg, alpha):
    kreg = "ct" if reg == "ct" else "c"
    a = adfuller(x, regression=reg, autolag="AIC")
    p = PhillipsPerron(x, trend=reg)
    k = kpss(x, regression=kreg, nlags="auto")
    cv = lambda d: [d["1%"], d["5%"], d["10%"]]
    rows = [
        ["ADF", a[0], a[1], *cv(a[4]), "جذر وحدة", "مستقرة" if a[1] < alpha else "غير مستقرة"],
        ["PP", p.stat, p.pvalue, *cv(p.critical_values), "جذر وحدة", "مستقرة" if p.pvalue < alpha else "غير مستقرة"],
        ["KPSS", k[0], k[1], k[3]["1%"], k[3]["5%"], k[3]["10%"], "مستقرة", "مستقرة" if k[1] >= alpha else "غير مستقرة"],
    ]
    df = pd.DataFrame(rows, columns=["الاختبار", "الإحصائية", "p-value", "CV 1%", "CV 5%", "CV 10%", "H0", "القرار"])
    return df, (a[1] < alpha, p.pvalue < alpha, k[1] >= alpha)


def verdict(flags):
    adf_s, pp_s, kp_s = flags
    if adf_s and pp_s and kp_s:
        return "success", "السلسلة **مستقرة** (الاختبارات الثلاثة متفقة)."
    if not adf_s and not pp_s and not kp_s:
        return "error", "السلسلة **غير مستقرة** (جذر وحدة) – جرّب الفروق d=1."
    return "warning", "**نتائج متضاربة**: غيّر مواصفة النموذج (ثابت / ثابت+اتجاه) أو جرّب الفروق، وتحقق من حجم العينة."


def page_lab():
    st.header("5) المخبر التطبيقي")
    src = st.radio("مصدر البيانات", ["محاكاة", "رفع ملف CSV"], horizontal=True)
    if src == "محاكاة":
        kind, n, seed = sidebar_sim("l", default=3)
        x = simulate(kind, n, seed)
    else:
        up = st.file_uploader("ارفع ملف CSV", type="csv")
        if up is None:
            st.stop()
        df = pd.read_csv(up)
        col = st.selectbox("العمود", df.select_dtypes("number").columns)
        x = df[col].dropna().reset_index(drop=True)
    c1, c2, c3 = st.columns(3)
    reg = c1.selectbox("مواصفة النموذج", ["c", "ct", "n"], format_func={"c": "ثابت", "ct": "ثابت + اتجاه", "n": "بدون"}.get)
    alpha = c2.selectbox("مستوى المعنوية α", [0.01, 0.05, 0.10], index=1)
    d = c3.selectbox("درجة الفروق d", [0, 1, 2])
    z = pd.Series(np.diff(x.values, d) if d else x.values)
    plot_diag(z, 20, f"d = {d}")
    try:
        res, flags = run_tests(z.values, reg, alpha)
    except Exception as e:  # noqa: BLE001
        st.error(f"تعذّر إجراء الاختبار: {e}")
        return
    st.dataframe(res.style.format({"الإحصائية": "{:.3f}", "p-value": "{:.4f}", "CV 1%": "{:.3f}", "CV 5%": "{:.3f}", "CV 10%": "{:.3f}"}), use_container_width=True)
    getattr(st, verdict(flags)[0])(verdict(flags)[1])
    st.caption("ملاحظة: KPSS لا يدعم مواصفة 'بدون'، فيُستعمل معها الثابت.")


PAGES = {
    "🏠 الرئيسية": page_home,
    "1) مفهوم الاستقرارية": page_concept,
    "2) أنواع الاستقرارية": page_types,
    "3) السيرورات العشوائية": page_stochastic,
    "4) اختبارات جذر الوحدة": page_tests_theory,
    "5) المخبر التطبيقي": page_lab,
}
choice = st.sidebar.radio("القائمة", list(PAGES))
PAGES[choice]()
