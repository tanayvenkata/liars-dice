"""
Statistical Utilities for Liar's Dice Analysis.

This module provides statistical functions for tournament analysis and
opponent modeling significance testing.

Functions:
    Confidence Intervals:
        wilson_ci(wins, n): 95% CI for proportions (better than normal approx)
        margin_of_error(wins, n): Half-width of CI
        format_win_rate_with_ci(wins, n): "67.0% ± 1.7%" format

    Significance Tests:
        binomial_test(k, n, p): Test if rate differs from baseline
        two_proportion_z_test(w1, n1, w2, n2): Compare two win rates
        chi_squared_test(w1, n1, w2, n2): Alternative comparison test

    Helpers:
        is_significantly_different(w1, n1, w2, n2): Quick boolean check
        is_significantly_better(w1, n1, w2, n2): One-sided test
        format_comparison(...): Human-readable comparison with stars

Dependencies:
    - scipy (optional but recommended): Provides exact tests
    - Falls back to normal approximation if scipy not available

Significance Markers:
    * p < 0.05, ** p < 0.01, *** p < 0.001

Example:
    from stats_utils import wilson_ci, is_significantly_better

    # 670 wins out of 1000 games
    lower, upper = wilson_ci(670, 1000)  # → (0.639, 0.699)

    # Compare two agents
    if is_significantly_better(670, 1000, 500, 1000):
        print("Agent 1 is significantly better!")
"""

from typing import Tuple
import math

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not installed, using approximate statistics")


def wilson_ci(wins: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Wilson score confidence interval for a proportion.

    Better than normal approximation for proportions near 0 or 1,
    and for small sample sizes.

    Args:
        wins: Number of successes
        n: Total trials
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        (lower, upper) bounds of confidence interval
    """
    if n == 0:
        return (0.0, 1.0)

    p = wins / n
    z = stats.norm.ppf(1 - (1 - confidence) / 2) if HAS_SCIPY else 1.96

    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator

    return (max(0.0, center - margin), min(1.0, center + margin))


def margin_of_error(wins: int, n: int, confidence: float = 0.95) -> float:
    """
    Calculate margin of error as half the confidence interval width.

    Args:
        wins: Number of successes
        n: Total trials
        confidence: Confidence level

    Returns:
        Margin of error (e.g., 0.018 for ±1.8%)
    """
    lower, upper = wilson_ci(wins, n, confidence)
    return (upper - lower) / 2


def format_win_rate_with_ci(wins: int, n: int, confidence: float = 0.95) -> str:
    """
    Format win rate with confidence interval.

    Args:
        wins: Number of wins
        n: Total games
        confidence: Confidence level

    Returns:
        Formatted string like "67.0% ± 1.7%"
    """
    if n == 0:
        return "N/A"

    rate = wins / n
    moe = margin_of_error(wins, n, confidence)
    return f"{rate:.1%} ± {moe:.1%}"


def binomial_test(successes: int, trials: int, null_p: float = 0.5,
                  alternative: str = 'two-sided') -> float:
    """
    Proper binomial test using scipy.

    Args:
        successes: Number of successes observed
        trials: Total number of trials
        null_p: Null hypothesis probability
        alternative: 'two-sided', 'greater', or 'less'

    Returns:
        p-value
    """
    if trials == 0:
        return 1.0

    if HAS_SCIPY:
        result = stats.binomtest(successes, trials, null_p, alternative=alternative)
        return result.pvalue
    else:
        # Fallback: normal approximation for large samples
        if trials < 30:
            return 1.0  # Not enough data for approximation

        observed_p = successes / trials
        se = math.sqrt(null_p * (1 - null_p) / trials)
        if se == 0:
            return 1.0

        z = abs(observed_p - null_p) / se
        # Rough two-sided p-value from z
        if z < 1.0:
            return 1.0
        elif z < 1.645:
            return 0.10
        elif z < 1.96:
            return 0.05
        elif z < 2.576:
            return 0.01
        else:
            return 0.001


def two_proportion_z_test(wins1: int, n1: int, wins2: int, n2: int) -> Tuple[float, float]:
    """
    Two-proportion z-test for comparing two win rates.

    Tests if two proportions are significantly different.

    Args:
        wins1: Wins for group 1
        n1: Total games for group 1
        wins2: Wins for group 2
        n2: Total games for group 2

    Returns:
        (z_statistic, p_value)
    """
    if n1 == 0 or n2 == 0:
        return (0.0, 1.0)

    p1 = wins1 / n1
    p2 = wins2 / n2

    # Pooled proportion
    p_pooled = (wins1 + wins2) / (n1 + n2)

    # Standard error
    se = math.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
    if se == 0:
        return (0.0, 1.0)

    z = (p1 - p2) / se

    if HAS_SCIPY:
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    else:
        # Approximate p-value
        abs_z = abs(z)
        if abs_z < 1.0:
            p_value = 1.0
        elif abs_z < 1.645:
            p_value = 0.10
        elif abs_z < 1.96:
            p_value = 0.05
        elif abs_z < 2.576:
            p_value = 0.01
        else:
            p_value = 0.001

    return (z, p_value)


def is_significantly_different(wins1: int, n1: int, wins2: int, n2: int,
                                alpha: float = 0.05) -> bool:
    """
    Test if two win rates are significantly different.

    Args:
        wins1: Wins for group 1
        n1: Total games for group 1
        wins2: Wins for group 2
        n2: Total games for group 2
        alpha: Significance level (default 0.05)

    Returns:
        True if significantly different at alpha level
    """
    _, p_value = two_proportion_z_test(wins1, n1, wins2, n2)
    return p_value < alpha


def is_significantly_better(wins1: int, n1: int, wins2: int, n2: int,
                             alpha: float = 0.05) -> bool:
    """
    Test if win rate 1 is significantly BETTER than win rate 2.

    One-sided test.

    Args:
        wins1: Wins for group 1 (the one we're testing is better)
        n1: Total games for group 1
        wins2: Wins for group 2
        n2: Total games for group 2
        alpha: Significance level (default 0.05)

    Returns:
        True if group 1 is significantly better at alpha level
    """
    if n1 == 0 or n2 == 0:
        return False

    p1 = wins1 / n1
    p2 = wins2 / n2

    # Must be actually better (not just testing for it)
    if p1 <= p2:
        return False

    z, two_sided_p = two_proportion_z_test(wins1, n1, wins2, n2)
    # One-sided p-value
    one_sided_p = two_sided_p / 2

    return one_sided_p < alpha


def format_comparison(name: str, wins1: int, n1: int, wins2: int, n2: int,
                       label1: str = "A", label2: str = "B") -> str:
    """
    Format a comparison between two win rates with significance.

    Args:
        name: Name of the comparison
        wins1, n1: Group 1 results
        wins2, n2: Group 2 results
        label1, label2: Labels for the groups

    Returns:
        Formatted string with rates and significance
    """
    rate1 = wins1 / n1 if n1 > 0 else 0
    rate2 = wins2 / n2 if n2 > 0 else 0
    diff = rate1 - rate2

    _, p_value = two_proportion_z_test(wins1, n1, wins2, n2)

    sig_marker = ""
    if p_value < 0.001:
        sig_marker = " ***"
    elif p_value < 0.01:
        sig_marker = " **"
    elif p_value < 0.05:
        sig_marker = " *"

    ci1 = format_win_rate_with_ci(wins1, n1)
    ci2 = format_win_rate_with_ci(wins2, n2)

    return (f"{name}: {label1}={ci1}, {label2}={ci2}, "
            f"diff={diff:+.1%}{sig_marker} (p={p_value:.3f})")


def chi_squared_test(wins1: int, n1: int, wins2: int, n2: int) -> float:
    """
    Chi-squared test for comparing two proportions.

    Alternative to z-test, especially for smaller samples.

    Args:
        wins1, n1: Group 1 results
        wins2, n2: Group 2 results

    Returns:
        p-value from chi-squared test
    """
    if not HAS_SCIPY:
        # Fall back to z-test
        _, p = two_proportion_z_test(wins1, n1, wins2, n2)
        return p

    # Create contingency table
    table = [[wins1, n1 - wins1], [wins2, n2 - wins2]]

    # Check for zeros (chi-squared doesn't handle well)
    if any(cell == 0 for row in table for cell in row):
        _, p = two_proportion_z_test(wins1, n1, wins2, n2)
        return p

    _, p_value, _, _ = stats.chi2_contingency(table)
    return p_value
