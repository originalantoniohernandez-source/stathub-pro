import numpy as np
from scipy import stats

class StatEngine:
    """Computation engine for StatHub Pro — covers descriptive, inferential, and distributional statistics."""

    VERSION = "2.2.0"

    # ==========================================================
    # ---------- Descriptive Statistics ----------
    # ==========================================================
    def mean(self, data):
        return np.mean(data)

    def median(self, data):
        return np.median(data)

    def mode(self, data):
        m = stats.mode(data, keepdims=True)
        return m.mode[0], m.count[0]

    def variance(self, data, sample=True):
        ddof = 1 if sample else 0
        return np.var(data, ddof=ddof)

    def std_dev(self, data, sample=True):
        ddof = 1 if sample else 0
        return np.std(data, ddof=ddof)

    def summary(self, data):
        """Return a dictionary summary of common descriptive measures."""
        return {
            "Count": len(data),
            "Mean": np.mean(data),
            "Median": np.median(data),
            "Mode": self.mode(data)[0],
            "Std Dev": np.std(data, ddof=1),
            "Variance": np.var(data, ddof=1),
            "Min": np.min(data),
            "Max": np.max(data),
            "Range": np.ptp(data),
            "IQR": stats.iqr(data)
        }

    # ==========================================================
    # ---------- Confidence Intervals ----------
    # ==========================================================
    def mean_confidence_interval(self, data, confidence=0.95):
        """Return (lower, upper, margin) for a mean confidence interval."""
        n = len(data)
        m = np.mean(data)
        s = stats.sem(data)
        t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
        margin = t_crit * s
        return m - margin, m + margin, margin

    def proportion_confidence_interval(self, p_hat, n, confidence=0.95):
        """Return (lower, upper, margin) for a proportion confidence interval."""
        z = stats.norm.ppf((1 + confidence) / 2)
        margin = z * np.sqrt((p_hat * (1 - p_hat)) / n)
        return p_hat - margin, p_hat + margin, margin

    # ==========================================================
    # ---------- Distributions ----------
    # ==========================================================
    # Normal
    def normal_pdf(self, x, mean=0, sd=1):
        return stats.norm.pdf(x, mean, sd)

    def normal_cdf(self, x, mean=0, sd=1):
        return stats.norm.cdf(x, mean, sd)

    def normal_inv(self, p, mean=0, sd=1):
        return stats.norm.ppf(p, mean, sd)

    # Student-t
    def t_pdf(self, x, df):
        return stats.t.pdf(x, df)

    def t_cdf(self, x, df):
        return stats.t.cdf(x, df)

    def t_inv(self, p, df):
        return stats.t.ppf(p, df)

    # Chi-square
    def chi2_pdf(self, x, df):
        return stats.chi2.pdf(x, df)

    def chi2_cdf(self, x, df):
        return stats.chi2.cdf(x, df)

    def chi2_inv(self, p, df):
        return stats.chi2.ppf(p, df)

    # F-distribution
    def f_pdf(self, x, df1, df2):
        return stats.f.pdf(x, df1, df2)

    def f_cdf(self, x, df1, df2):
        return stats.f.cdf(x, df1, df2)

    def f_inv(self, p, df1, df2):
        return stats.f.ppf(p, df1, df2)

    # Binomial
    def binom_pmf(self, k, n, p):
        return stats.binom.pmf(k, n, p)

    def binom_cdf(self, k, n, p):
        return stats.binom.cdf(k, n, p)

    # Poisson
    def poisson_pmf(self, k, lam):
        return stats.poisson.pmf(k, lam)

    def poisson_cdf(self, k, lam):
        return stats.poisson.cdf(k, lam)

    # Exponential
    def expon_cdf(self, x, rate=1):
        return stats.expon.cdf(x, scale=1/rate)

    def expon_inv(self, p, rate=1):
        return stats.expon.ppf(p, scale=1/rate)

    # Uniform
    def uniform_cdf(self, x, a=0, b=1):
        return stats.uniform.cdf(x, a, b - a)

    def uniform_inv(self, p, a=0, b=1):
        return stats.uniform.ppf(p, a, b - a)

    # ==========================================================
    # ---------- Hypothesis Tests ----------
    # ==========================================================
    def one_sample_ttest(self, data, mu0):
        """Perform one-sample t-test."""
        return stats.ttest_1samp(data, mu0)

    def two_sample_ttest(self, data1, data2, equal_var=True):
        """Independent two-sample t-test."""
        return stats.ttest_ind(data1, data2, equal_var=equal_var)

    def paired_ttest(self, before, after):
        """Paired-sample t-test."""
        return stats.ttest_rel(before, after)

    def one_sample_ztest(self, data, mu0, sigma):
        """Z-test for population mean (σ known)."""
        n = len(data)
        xbar = np.mean(data)
        z = (xbar - mu0) / (sigma / np.sqrt(n))
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        return z, p

    def one_proportion_ztest(self, x, n, p0):
        """Z-test for single population proportion."""
        phat = x / n
        z = (phat - p0) / np.sqrt(p0 * (1 - p0) / n)
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        return z, p

    def two_proportion_ztest(self, x1, n1, x2, n2):
        """Z-test for two population proportions."""
        p1, p2 = x1 / n1, x2 / n2
        p_pool = (x1 + x2) / (n1 + n2)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
        z = (p1 - p2) / se
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        return z, p

    # ==========================================================
    # ---------- Regression ----------
    # ==========================================================
    def simple_regression(self, x, y):
        """Perform simple linear regression and return key statistics."""
        slope, intercept, r, p, stderr = stats.linregress(x, y)
        return {
            "Slope": slope,
            "Intercept": intercept,
            "Correlation (r)": r,
            "R²": r**2,
            "p-value": p,
            "Std Error": stderr
        }
