import streamlit as st
import numpy as np
import plotly.graph_objects as go
from statengine import StatEngine
from scipy import stats

# =====================================================
# ---------- Helper Plot Functions ----------
# =====================================================
def plot_normal_pvalue(z, alt):
    xs = np.linspace(-4, 4, 800)
    ys = stats.norm.pdf(xs)
    fig = go.Figure([go.Scatter(x=xs, y=ys, mode="lines", name="N(0,1)")])
    if alt == "≠":
        mask = (xs <= -abs(z)) | (xs >= abs(z))
    elif alt == ">":
        mask = xs >= z
    else:
        mask = xs <= z
    fig.add_trace(go.Scatter(x=xs[mask], y=ys[mask], fill="tozeroy",
                             name="p-value area",
                             fillcolor="rgba(0,120,255,0.35)"))
    fig.update_layout(height=350, xaxis_title="z", yaxis_title="Density",
                      showlegend=False)
    return fig


def plot_t_pvalue(t, df, alt):
    xs = np.linspace(-5, 5, 800)
    ys = stats.t.pdf(xs, df)
    fig = go.Figure([go.Scatter(x=xs, y=ys, mode="lines", name=f"t(df={df})")])
    if alt == "≠":
        mask = (xs <= -abs(t)) | (xs >= abs(t))
    elif alt == ">":
        mask = xs >= t
    else:
        mask = xs <= t
    fig.add_trace(go.Scatter(x=xs[mask], y=ys[mask], fill="tozeroy",
                             name="p-value area",
                             fillcolor="rgba(255,120,0,0.35)"))
    fig.update_layout(height=350, xaxis_title="t", yaxis_title="Density",
                      showlegend=False)
    return fig


# =====================================================
# ---------- Streamlit App ----------
# =====================================================
st.set_page_config(page_title="StatHub Pro", layout="wide")
st.title("📊 StatHub Pro — Statistics Workbench")
st.caption(f"Powered by StatEngine v{StatEngine.VERSION}")

stat = StatEngine()

# Sidebar navigation
section = st.sidebar.radio("Select Section 📚", [
    "Descriptive Statistics",
    "Distribution Calculators",
    "Inference Tests",
    "P-Value Finder"
])

# =====================================================
# ---------- Descriptive Statistics ----------
# =====================================================
if section == "Descriptive Statistics":
    st.header("Descriptive Statistics Summary")
    raw = st.text_area("Enter data (comma or space separated):", "4, 5, 6, 7, 8")
    try:
        data = np.array([float(x) for x in raw.replace(",", " ").split()])
        stats_dict = stat.summary(data)
        st.subheader("Summary Results")
        st.json(stats_dict)
        ci_low, ci_high, margin = stat.mean_confidence_interval(data)
        st.write(f"95% CI for mean: ({ci_low:.3f}, {ci_high:.3f})   ± {margin:.3f}")
    except Exception as e:
        st.error(f"Error processing data: {e}")

# =====================================================
# ---------- Distribution Calculators ----------
# =====================================================
elif section == "Distribution Calculators":
    dist = st.selectbox("Choose Distribution:", [
        "Normal", "t", "Chi-square", "Binomial", "Poisson", "Exponential", "Uniform"
    ])
    if dist == "Normal":
        x = st.number_input("x value:", value=0.0)
        mean = st.number_input("Mean (μ):", value=0.0)
        sd = st.number_input("Std Dev (σ):", value=1.0, min_value=0.0001)
        st.write(f"PDF = {stat.normal_pdf(x, mean, sd):.5f}")
        st.write(f"CDF = {stat.normal_cdf(x, mean, sd):.5f}")

    elif dist == "t":
        x = st.number_input("t value:", value=1.0)
        df = st.number_input("df:", value=10)
        st.write(f"PDF = {stat.t_pdf(x, df):.5f}")
        st.write(f"CDF = {stat.t_cdf(x, df):.5f}")

    elif dist == "Chi-square":
        x = st.number_input("x:", value=2.0)
        df = st.number_input("df:", value=5)
        st.write(f"PDF = {stats.chi2.pdf(x, df):.5f}")
        st.write(f"CDF = {stat.chi2_cdf(x, df):.5f}")

    elif dist == "Binomial":
        k = st.number_input("k (successes):", value=3)
        n = st.number_input("n (trials):", value=10)
        p = st.number_input("p (success prob):", value=0.5)
        st.write(f"PMF = {stat.binom_pmf(k, n, p):.5f}")
        st.write(f"CDF = {stat.binom_cdf(k, n, p):.5f}")

    elif dist == "Poisson":
        k = st.number_input("k:", value=2)
        lam = st.number_input("λ (rate):", value=3.0)
        st.write(f"PMF = {stat.poisson_pmf(k, lam):.5f}")
        st.write(f"CDF = {stat.poisson_cdf(k, lam):.5f}")

    elif dist == "Exponential":
        x = st.number_input("x:", value=1.0)
        rate = st.number_input("Rate (λ):", value=1.0)
        st.write(f"CDF = {stat.expon_cdf(x, rate):.5f}")

    elif dist == "Uniform":
        x = st.number_input("x:", value=0.5)
        a = st.number_input("a:", value=0.0)
        b = st.number_input("b:", value=1.0)
        st.write(f"CDF = {stat.uniform_cdf(x, a, b):.5f}")

# =====================================================
# ---------- Inference Tests ----------
# =====================================================
elif section == "Inference Tests":
    st.header("Inference and Hypothesis Testing")

    test = st.selectbox("Choose Test:", [
        "One-sample z-test",
        "One-sample t-test",
        "Two-sample t-test",
        "Paired t-test",
        "One-proportion z-test",
        "Two-proportion z-test"
    ])

    alt = st.radio("Alternative Hypothesis (Hₐ):",
                   ["≠ (two-tailed)", "> (right-tailed)", "< (left-tailed)"])
    alpha = st.number_input("Significance Level (α):", value=0.05, min_value=0.001, max_value=0.5, step=0.001)

    # -------------------- ONE-SAMPLE T --------------------
    if test == "One-sample t-test":
        mode = st.radio("Input Mode:", ["Raw data", "Summary stats"])
        if mode == "Raw data":
            data = st.text_area("Enter sample data:", "88, 90, 85, 92, 87, 91, 89")
            mu0 = st.number_input("Population mean μ₀:", value=85.0)
            if st.button("Run t-test"):
                data = np.array([float(x) for x in data.replace(",", " ").split()])
                t, p = stat.one_sample_ttest(data, mu0)
                st.success(f"t = {t:.4f}   p = {p:.4f}")
                st.plotly_chart(plot_t_pvalue(t, len(data)-1, alt[0]), use_container_width=True)
        else:
            mean = st.number_input("Sample mean:", value=88.0)
            sd = st.number_input("Sample SD:", value=5.0)
            n = st.number_input("Sample size n:", value=30)
            mu0 = st.number_input("Population mean μ₀:", value=85.0)
            if st.button("Run summary t-test"):
                t = (mean - mu0) / (sd / np.sqrt(n))
                df = n - 1
                if "≠" in alt:
                    p = 2 * (1 - stats.t.cdf(abs(t), df))
                elif ">" in alt:
                    p = 1 - stats.t.cdf(t, df)
                else:
                    p = stats.t.cdf(t, df)
                st.success(f"t = {t:.4f}   p = {p:.4f}")
                st.plotly_chart(plot_t_pvalue(t, df, alt[0]), use_container_width=True)

    # -------------------- ONE-SAMPLE Z --------------------
    elif test == "One-sample z-test":
        xbar = st.number_input("Sample mean x̄:", value=90.0)
        mu0 = st.number_input("Population mean μ₀:", value=85.0)
        sigma = st.number_input("Population σ:", value=5.0)
        n = st.number_input("Sample size n:", value=25)
        if st.button("Run z-test"):
            z = (xbar - mu0) / (sigma / np.sqrt(n))
            if "≠" in alt:
                p = 2 * (1 - stats.norm.cdf(abs(z)))
            elif ">" in alt:
                p = 1 - stats.norm.cdf(z)
            else:
                p = stats.norm.cdf(z)
            st.success(f"z = {z:.4f}   p = {p:.4f}")
            st.plotly_chart(plot_normal_pvalue(z, alt[0]), use_container_width=True)

    # -------------------- TWO-SAMPLE T --------------------
    elif test == "Two-sample t-test":
        m1 = st.number_input("Mean 1:", value=85.0)
        s1 = st.number_input("SD 1:", value=4.0)
        n1 = st.number_input("n₁:", value=25)
        m2 = st.number_input("Mean 2:", value=82.0)
        s2 = st.number_input("SD 2:", value=5.0)
        n2 = st.number_input("n₂:", value=25)
        equal_var = st.checkbox("Assume equal variances", value=True)
        if st.button("Run 2-sample t-test"):
            se = np.sqrt((s1**2)/n1 + (s2**2)/n2)
            df = int(n1 + n2 - 2)
            t = (m1 - m2) / se
            if "≠" in alt:
                p = 2 * (1 - stats.t.cdf(abs(t), df))
            elif ">" in alt:
                p = 1 - stats.t.cdf(t, df)
            else:
                p = stats.t.cdf(t, df)
            st.success(f"t = {t:.4f}   p = {p:.4f}")
            st.plotly_chart(plot_t_pvalue(t, df, alt[0]), use_container_width=True)

    # -------------------- PROPORTION TESTS --------------------
    elif test == "One-proportion z-test":
        x = st.number_input("Successes (x):", value=64)
        n = st.number_input("Sample size (n):", value=100)
        p0 = st.number_input("Hypothesized p₀:", value=0.5)
        if st.button("Run prop z-test"):
            z, p = stat.one_proportion_ztest(x, n, p0)
            st.success(f"z = {z:.4f}   p = {p:.4f}")
            st.plotly_chart(plot_normal_pvalue(z, alt[0]), use_container_width=True)

    elif test == "Two-proportion z-test":
        x1 = st.number_input("Group 1 successes:", value=64)
        n1 = st.number_input("Group 1 n:", value=100)
        x2 = st.number_input("Group 2 successes:", value=56)
        n2 = st.number_input("Group 2 n:", value=100)
        if st.button("Run 2-prop z-test"):
            z, p = stat.two_proportion_ztest(x1, n1, x2, n2)
            st.success(f"z = {z:.4f}   p = {p:.4f}")
            st.plotly_chart(plot_normal_pvalue(z, alt[0]), use_container_width=True)

    # -------------------- PAIRED T --------------------
    elif test == "Paired t-test":
        before = st.text_area("Before data:", "85, 88, 90, 92, 91")
        after = st.text_area("After data:", "88, 90, 93, 95, 92")
        if st.button("Run paired t-test"):
            b = np.array([float(x) for x in before.replace(",", " ").split()])
            a = np.array([float(x) for x in after.replace(",", " ").split()])
            t, p = stat.paired_ttest(b, a)
            st.success(f"t = {t:.4f}   p = {p:.4f}")
            st.plotly_chart(plot_t_pvalue(t, len(b)-1, alt[0]), use_container_width=True)

# =====================================================
# ---------- UNIVERSAL P-VALUE FINDER ----------
# =====================================================
elif section == "P-Value Finder":
    st.header("🎯 Universal P-Value Finder (All Tail Results + Critical Values)")

    dist = st.selectbox("Select test distribution:",
        ["Z (Normal)", "T (Student)", "Chi-square (χ²)", "F (Variance Ratio)"])

    test_stat = st.number_input("Enter test statistic value:", value=0.0, step=0.1)
    alpha = st.number_input("Significance Level (α):", value=0.05, min_value=0.0001, max_value=0.5)
    selected_tail = st.radio("Select test direction:", ["< (Left-tailed)", "> (Right-tailed)", "≠ (Two-tailed)"])

    # compute p-values and critical values
    if dist == "Z (Normal)":
        p_left, p_right = stats.norm.cdf(test_stat), 1 - stats.norm.cdf(test_stat)
        p_two = 2 * (1 - stats.norm.cdf(abs(test_stat)))
        z_crit_left = stats.norm.ppf(alpha)
        z_crit_right = stats.norm.ppf(1 - alpha)
        z_crit_two = stats.norm.ppf(1 - alpha/2)
        crits = {"left": z_crit_left, "right": z_crit_right, "two": z_crit_two}

    elif dist == "T (Student)":
        df = st.number_input("Degrees of freedom (df):", min_value=1, value=10)
        p_left, p_right = stats.t.cdf(test_stat, df), 1 - stats.t.cdf(test_stat, df)
        p_two = 2 * (1 - stats.t.cdf(abs(test_stat), df))
        crits = {
            "left": stats.t.ppf(alpha, df),
            "right": stats.t.ppf(1 - alpha, df),
            "two": stats.t.ppf(1 - alpha / 2, df)
        }

    elif dist == "Chi-square (χ²)":
        df = st.number_input("Degrees of freedom (df):", min_value=1, value=5)
        p_left, p_right = stats.chi2.cdf(test_stat, df), 1 - stats.chi2.cdf(test_stat, df)
        p_two = min(1.0, 2 * min(p_left, p_right))
        crits = {
            "left": stats.chi2.ppf(alpha, df),
            "right": stats.chi2.ppf(1 - alpha, df),
            "two": (stats.chi2.ppf(alpha/2, df), stats.chi2.ppf(1 - alpha/2, df))
        }

    elif dist == "F (Variance Ratio)":
        df1 = st.number_input("Numerator df (df₁):", min_value=1, value=5)
        df2 = st.number_input("Denominator df (df₂):", min_value=1, value=5)
        p_left, p_right = stats.f.cdf(test_stat, df1, df2), 1 - stats.f.cdf(test_stat, df1, df2)
        p_two = min(1.0, 2 * min(p_left, p_right))
        crits = {
            "left": stats.f.ppf(alpha, df1, df2),
            "right": stats.f.ppf(1 - alpha, df1, df2),
            "two": (stats.f.ppf(alpha/2, df1, df2), stats.f.ppf(1 - alpha/2, df1, df2))
        }

    chosen = "left" if "<" in selected_tail else "right" if ">" in selected_tail else "two"
    p_values = {"left": p_left, "right": p_right, "two": p_two}

    cols = st.columns(3)
    for i, tail in enumerate(["left", "right", "two"]):
        with cols[i]:
            highlight = "🟩" if tail == chosen else "⬜️"
            st.markdown(f"### {highlight} {tail.capitalize()}-tailed Test")
            st.markdown(f"**p-value:** `{p_values[tail]:.4f}`")
            st.markdown(f"**Critical Value(s):** `{crits[tail]}`")

    # Plot shaded areas for Z/T
    if dist.startswith("Z") or dist.startswith("T"):
        x = np.linspace(-4, 4, 500)
        y = stats.norm.pdf(x) if dist.startswith("Z") else stats.t.pdf(x, df)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"{dist} PDF"))
        fig.add_vline(x=test_stat, line_color="red", line_dash="dash", annotation_text="Test Stat")
        st.plotly_chart(fig, use_container_width=True)
