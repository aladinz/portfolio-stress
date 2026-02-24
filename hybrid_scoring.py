"""
hybrid_scoring.py
=================
Hybrid Scoring System and Portfolio Categorization System
for the Portfolio Stress Engine.

Five-pillar weighted scoring model (total: 0-100, letter grade A+ → F):
  1. Structural Integrity   (30%)  — diversification, concentration, correlation
  2. Macro-Resilience       (25%)  — drawdown depth, crisis-window behavior
  3. Intent Alignment       (20%)  — asset-class breadth, global reach, income coverage
  4. Volatility Appropriateness (15%) — annualised vol vs risk levels
  5. Portfolio Health       (10%)  — Sharpe quality, tail-risk ratio, win-rate

Portfolio categories: Conservative · Moderate · Growth · Aggressive · Resilient
"""

# ── Known-ticker lookup tables ────────────────────────────────────────────────

def _clamp(score, lo=0, hi=100):
    """Clamp a numeric score into the [lo, hi] range (default 0-100).
    Applied at the exit of every pillar function so that unexpected
    edge-case inputs never produce an out-of-range result."""
    return max(lo, min(hi, float(score)))

# Fixed-income / bond ETFs
_BOND_TICKERS = {
    "BND", "AGG", "TLT", "IEF", "SHY", "VGIT", "VGSH", "VGLT", "BIL",
    "LQD", "HYG", "BNDX", "IAGG", "BNDW", "BSV", "BIV", "BLV", "VCSH",
    "VCIT", "VCLT", "MBB", "EMB", "NEAR", "FLOT", "GVI", "SCHZ", "FXNAX",
    "FBND", "FALN", "ISTB", "IUSB", "IUSIG", "IGIB", "SPSB", "SPIB",
    "SPLB", "SPTS", "SPTIG", "FLRN", "STIP", "LTPZ", "TIPX",
}

# International (non-US) equity ETFs
_INTL_TICKERS = {
    "VXUS", "VEU", "VEA", "EFA", "IEFA", "EEM", "VWO", "IEMG", "ACWX",
    "SPDW", "SPEM", "FNDF", "SCZ", "EFAV", "EWJ", "EWG", "EWY", "EWZ",
    "ACIM", "GEM", "FZILX", "FSPSX",
}

# Inflation-hedge assets (TIPS, real-estate, commodities)
_INFLATION_TICKERS = {
    "GLD", "IAU", "SLV", "PDBC", "DJP", "GSG", "TIPS", "TIP", "VNQ",
    "IYR", "SCHH", "REZ", "REET", "USRT", "INFL", "RINF", "LTPZ", "TIPX",
}

# Dividend / income-oriented equity ETFs
_INCOME_TICKERS = {
    "SCHD", "VIG", "DVY", "VYM", "HDV", "NOBL", "SDY", "DGRO", "DGRW",
    "PFF", "FPE", "SPYD", "RDIV", "IDV", "SCHY", "FDL", "DLN", "CDC", "COWZ",
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _classify_holdings(tickers, weights_vec):
    """
    Tag each holding by asset class and return portfolio-level allocation
    fractions for: equity, bond, international, inflation, income.
    A holding can belong to multiple classes (e.g. TIPS = bond + inflation).
    Equity = residual after removing pure bond and inflation holdings.
    """
    bond_w = intl_w = infl_w = income_w = 0.0
    for t, w in zip(tickers, weights_vec):
        tu = t.upper()
        if tu in _BOND_TICKERS:
            bond_w += float(w)
        if tu in _INTL_TICKERS:
            intl_w += float(w)
        if tu in _INFLATION_TICKERS:
            infl_w += float(w)
        if tu in _INCOME_TICKERS:
            income_w += float(w)

    # equity = everything not classified as pure bond/inflation
    equity_w = max(0.0, 1.0 - bond_w - max(0.0, infl_w - bond_w))
    return {
        "equity":        equity_w,
        "bond":          bond_w,
        "international": intl_w,
        "inflation":     infl_w,
        "income":        income_w,
    }


def _hhi(weights_vec):
    """Herfindahl-Hirschman Index: sum of squared weights. Range 1/n → 1."""
    return float(sum(w ** 2 for w in weights_vec))


def _avg_pairwise_corr(corr_matrix, tickers):
    """Average of all off-diagonal correlation values."""
    n = len(tickers)
    if n <= 1 or corr_matrix is None:
        return 0.5  # neutral default

    available = [t for t in tickers if t in corr_matrix.columns]
    if len(available) <= 1:
        return 0.5

    sub = corr_matrix.loc[available, available]
    vals = [
        sub.iloc[r, c]
        for r in range(len(available))
        for c in range(len(available))
        if r != c
    ]
    import math
    clean = [v for v in vals if not math.isnan(v)]
    return float(sum(clean) / len(clean)) if clean else 0.5


# ── Pillar 1 – Structural Integrity (weight 30%) ──────────────────────────────

def score_structural_integrity(tickers, weights_vec, corr_matrix):
    """
    Measures how well the portfolio is built:
      A) Number of distinct holdings                     (max 30 pts)
      B) Concentration via Herfindahl-Hirschman Index    (max 40 pts)
      C) Average pairwise correlation (lower = better)   (max 30 pts)

    Returns (score 0-100, explanation_string)
    """
    n = len(tickers)

    # A) Holding count — more holdings → better diversification
    n_score = min(30, n * 4)  # 4 pts each, caps at 30 (i.e. 8+ holdings)

    # B) HHI — lower concentration is better
    hhi = _hhi(weights_vec)
    if   hhi < 0.08: hhi_score = 40
    elif hhi < 0.12: hhi_score = 32
    elif hhi < 0.18: hhi_score = 24
    elif hhi < 0.25: hhi_score = 15
    elif hhi < 0.35: hhi_score = 8
    else:            hhi_score = 3

    # C) Average correlation — lower means holdings move more independently
    avg_corr = _avg_pairwise_corr(corr_matrix, tickers)
    if   avg_corr < 0.35: corr_score = 30
    elif avg_corr < 0.50: corr_score = 22
    elif avg_corr < 0.65: corr_score = 14
    elif avg_corr < 0.80: corr_score = 7
    else:                  corr_score = 2

    score = n_score + hhi_score + corr_score  # 0-100

    # Build a concise explanation
    if n < 5:
        n_note = f"{n} holdings — low diversification"
    elif n < 8:
        n_note = f"{n} holdings — moderate diversification"
    else:
        n_note = f"{n} holdings — well spread"

    if hhi < 0.12:
        hhi_note = "low concentration (HHI)"
    elif hhi < 0.25:
        hhi_note = f"moderate concentration (HHI {hhi:.2f})"
    else:
        hhi_note = f"high concentration (HHI {hhi:.2f})"

    if avg_corr < 0.50:
        corr_note = f"low avg correlation ({avg_corr:.2f})"
    elif avg_corr < 0.70:
        corr_note = f"moderate avg correlation ({avg_corr:.2f})"
    else:
        corr_note = f"high avg correlation ({avg_corr:.2f}) — holdings move together"

    explanation = f"{n_note}; {hhi_note}; {corr_note}"
    return _clamp(score), explanation


# ── Pillar 2 – Macro-Resilience (weight 25%) ─────────────────────────────────

def score_macro_resilience(full_stats, stress_results):
    """
    Measures how the portfolio withstands macro shocks:
      A) Full-period max drawdown                      (max 40 pts)
      B) Average Sharpe ratio across stress windows    (max 30 pts)
      C) Average total return across stress windows    (max 30 pts)

    stress_results: list of (label, start, end, stats_dict | None)
    Returns (score 0-100, explanation_string)
    """
    max_dd = full_stats["max_dd"]

    # A) Full-period drawdown — shallower is better
    if   max_dd >= -0.10: dd_score = 40
    elif max_dd >= -0.15: dd_score = 34
    elif max_dd >= -0.20: dd_score = 28
    elif max_dd >= -0.25: dd_score = 22
    elif max_dd >= -0.30: dd_score = 16
    elif max_dd >= -0.40: dd_score = 9
    else:                  dd_score = 3

    # B & C) Average across available stress windows
    valid = [s for _, _, _, s in stress_results if s is not None]
    if valid:
        avg_stress_sharpe = sum(s["sharpe"] for s in valid) / len(valid)
        avg_stress_return = sum(s["total_return"] for s in valid) / len(valid)
    else:
        avg_stress_sharpe = 0.0
        avg_stress_return = 0.0

    # B) Stress Sharpe quality
    if   avg_stress_sharpe > 0.0:  sharpe_score = 30
    elif avg_stress_sharpe > -0.5: sharpe_score = 20
    elif avg_stress_sharpe > -1.0: sharpe_score = 12
    elif avg_stress_sharpe > -1.5: sharpe_score = 5
    else:                           sharpe_score = 2

    # C) Stress return depth — how badly did the portfolio get hit?
    if   avg_stress_return > -0.05:  ret_score = 30
    elif avg_stress_return > -0.10:  ret_score = 24
    elif avg_stress_return > -0.15:  ret_score = 17
    elif avg_stress_return > -0.20:  ret_score = 11
    elif avg_stress_return > -0.30:  ret_score = 6
    else:                             ret_score = 2

    score = dd_score + sharpe_score + ret_score  # 0-100

    dd_note = f"Max DD {max_dd:.1%}"
    if valid:
        stress_note = (
            f"avg crisis return {avg_stress_return:.1%}, "
            f"avg crisis Sharpe {avg_stress_sharpe:.2f}"
        )
    else:
        stress_note = "no overlapping stress window data"

    return _clamp(score), f"{dd_note}; {stress_note}"


# ── Pillar 3 – Intent Alignment (weight 20%) ─────────────────────────────────

def score_intent_alignment(tickers, weights_vec):
    """
    Measures how coherently the portfolio covers its apparent investment intent:
      A) Asset-class balance (equity vs bond mix)       (max 50 pts)
      B) Global diversification (international exposure)(max 30 pts)
      C) Defensive / income coverage                    (max 20 pts)

    Returns (score 0-100, explanation_string)
    """
    alloc = _classify_holdings(tickers, weights_vec)
    bond   = alloc["bond"]
    intl   = alloc["international"]
    income = alloc["income"]
    equity = alloc["equity"]

    # A) Asset-class balance — reward sensible equity/bond mix
    if   0.10 <= bond <= 0.20:  balance_score = 50   # light bond buffer
    elif 0.20 <  bond <= 0.40:  balance_score = 46   # classic balanced
    elif 0.40 <  bond <= 0.60:  balance_score = 40   # conservative balanced
    elif bond > 0.60:            balance_score = 34  # bond-heavy
    elif equity >= 0.90:         balance_score = 36  # pure equity, no cushion
    else:                        balance_score = 42  # mostly equity with some bond

    # B) International exposure — reduces home-country risk
    if   intl >= 0.25: intl_score = 30
    elif intl >= 0.15: intl_score = 24
    elif intl >= 0.08: intl_score = 16
    elif intl >= 0.03: intl_score = 8
    else:              intl_score = 2

    # C) Income / inflation coverage — defensive tilt
    income_flag = income >= 0.05
    inflation_flag = alloc["inflation"] >= 0.03
    def_score = 10 * income_flag + 10 * inflation_flag  # 0, 10, or 20

    score = balance_score + intl_score + def_score  # 0-100

    notes = []
    notes.append(f"equity {equity:.0%} / bond {bond:.0%}")
    if intl >= 0.08:
        notes.append(f"global diversification ({intl:.0%} intl)")
    else:
        notes.append("limited international exposure")
    if income_flag:
        notes.append("income tilt present")
    if inflation_flag:
        notes.append("inflation hedge present")
    if not income_flag and not inflation_flag:
        notes.append("no dedicated income or inflation hedge")

    return _clamp(score), "; ".join(notes)


# ── Pillar 4 – Volatility Appropriateness (weight 15%) ───────────────────────

def score_volatility_appropriateness(full_stats):
    """
    Rewards portfolios whose volatility is in a reasonable range.
    Very high volatility drags down the score; very low (if Sharpe is also
    decent) is rewarded.
      - Base score from vol level (max 70 pts)
      - Sharpe-compensation bonus: high vol forgiven if Sharpe is strong (max 30 pts)

    Returns (score 0-100, explanation_string)
    """
    vol    = full_stats["vol"]
    sharpe = full_stats["sharpe"]

    # Base score — tiered by annualised volatility
    if   vol <= 0.10: base = 70
    elif vol <= 0.14: base = 60
    elif vol <= 0.17: base = 50
    elif vol <= 0.20: base = 40
    elif vol <= 0.23: base = 30
    elif vol <= 0.27: base = 20
    else:             base = 10

    # Sharpe-compensation: if taking lots of vol, are we getting paid for it?
    if   sharpe >= 1.2: compensation = 30
    elif sharpe >= 0.8: compensation = 22
    elif sharpe >= 0.5: compensation = 14
    elif sharpe >= 0.2: compensation = 6
    else:               compensation = 0

    score = min(100, base + compensation)

    vol_note    = f"annualised vol {vol:.1%}"
    sharpe_note = f"Sharpe {sharpe:.2f} ({'compensates' if compensation >= 14 else 'does not fully compensate'})"
    return _clamp(score), f"{vol_note}; {sharpe_note}"


# ── Pillar 5 – Portfolio Health (weight 10%) ──────────────────────────────────

def score_portfolio_health(full_stats, portfolio_returns):
    """
    Three health indicators:
      A) Sharpe quality                             (max 50 pts)
      B) Tail-risk ratio CVaR/VaR (lower = thinner tail) (max 25 pts)
      C) Daily win-rate (% of positive return days)(max 25 pts)

    Returns (score 0-100, explanation_string)
    """
    s      = full_stats["sharpe"]
    var95  = full_stats["var_95"]
    cvar95 = full_stats["cvar_95"]

    # A) Sharpe ratio
    if   s >= 1.5: sharpe_pts = 50
    elif s >= 1.0: sharpe_pts = 42
    elif s >= 0.7: sharpe_pts = 33
    elif s >= 0.4: sharpe_pts = 22
    elif s >= 0.0: sharpe_pts = 10
    else:          sharpe_pts = 0

    # B) Fat-tail ratio: cvar / var — both negative, so ratio > 1 = fatter tail
    if var95 != 0:
        fat_ratio = cvar95 / var95  # both negative → ratio > 1 means worse tails
        if   fat_ratio <= 1.25: tail_pts = 25
        elif fat_ratio <= 1.40: tail_pts = 20
        elif fat_ratio <= 1.55: tail_pts = 14
        elif fat_ratio <= 1.75: tail_pts = 8
        else:                   tail_pts = 3
    else:
        fat_ratio = 1.0
        tail_pts  = 10

    # C) Win-rate: % of daily returns that are positive
    if portfolio_returns is not None and len(portfolio_returns) > 0:
        win_rate = float((portfolio_returns > 0).mean())
    else:
        win_rate = 0.5

    if   win_rate >= 0.55: wr_pts = 25
    elif win_rate >= 0.53: wr_pts = 21
    elif win_rate >= 0.51: wr_pts = 17
    elif win_rate >= 0.49: wr_pts = 12
    elif win_rate >= 0.47: wr_pts = 7
    else:                  wr_pts = 2

    score = sharpe_pts + tail_pts + wr_pts  # 0-100

    explanation = (
        f"Sharpe {s:.2f}; "
        f"tail ratio {fat_ratio:.2f} ({'thin' if fat_ratio <= 1.4 else 'fat'} tails); "
        f"win-rate {win_rate:.1%}"
    )
    return _clamp(score), explanation


# ── Final scoring function ────────────────────────────────────────────────────

# Pillar weights (must sum to 1.0)
PILLAR_WEIGHTS = {
    "Structural Integrity":       0.30,
    "Macro-Resilience":           0.25,
    "Intent Alignment":           0.20,
    "Volatility Appropriateness": 0.15,
    "Portfolio Health":           0.10,
}

# Letter-grade boundaries (total weighted score 0–100)
_GRADE_BANDS = [
    (92, "A+", "#10b981"),
    (84, "A",  "#10b981"),
    (76, "A-", "#34d399"),
    (68, "B+", "#6366f1"),
    (60, "B",  "#6366f1"),
    (52, "B-", "#818cf8"),
    (44, "C+", "#f59e0b"),
    (36, "C",  "#f59e0b"),
    (28, "C-", "#fbbf24"),
    (15, "D",  "#ef4444"),
    ( 0, "F",  "#ef4444"),
]


def _letter_grade(score):
    for threshold, letter, color in _GRADE_BANDS:
        if score >= threshold:
            return letter, color
    return "F", "#ef4444"


def _grade_explanation(pillar_scores, letter):
    """
    Build a one-sentence summary highlighting the strongest and weakest pillars.
    """
    best  = max(pillar_scores, key=lambda p: p["weighted"])
    worst = min(pillar_scores, key=lambda p: p["weighted"])
    verb  = {
        "A+": "exceptional", "A":  "strong",   "A-": "solid",
        "B+": "above-average","B": "decent",   "B-": "mixed",
        "C+": "mediocre",    "C":  "weak",      "C-": "poor",
        "D":  "very poor",   "F":  "critically weak",
    }.get(letter, "mixed")
    return (
        f"Overall {verb} — strongest in {best['pillar']} "
        f"({best['raw']:.0f}/100), "
        f"most room to improve in {worst['pillar']} "
        f"({worst['raw']:.0f}/100)."
    )


def compute_hybrid_score(tickers, weights_vec, full_stats,
                          stress_results, corr_matrix, portfolio_returns):
    """
    Main entry point — computes all five pillars, applies weights,
    and returns a comprehensive result dict.

    Returns:
      {
        "pillar_scores":   list of dicts  (pillar, raw, weight, weighted, explanation)
        "total_score":     float 0-100
        "letter_grade":    str  e.g. "B+"
        "grade_color":     hex  colour for the grade
        "explanation":     str  one-sentence summary
      }
    """
    raw_scores = {
        "Structural Integrity":       score_structural_integrity(
                                          tickers, weights_vec, corr_matrix),
        "Macro-Resilience":           score_macro_resilience(
                                          full_stats, stress_results),
        "Intent Alignment":           score_intent_alignment(
                                          tickers, weights_vec),
        "Volatility Appropriateness": score_volatility_appropriateness(
                                          full_stats),
        "Portfolio Health":           score_portfolio_health(
                                          full_stats, portfolio_returns),
    }

    pillar_scores = []
    total = 0.0
    for pillar, weight in PILLAR_WEIGHTS.items():
        raw, expl = raw_scores[pillar]
        weighted  = raw * weight
        total    += weighted
        pillar_scores.append({
            "pillar":      pillar,
            "raw":         raw,
            "weight":      weight,
            "weighted":    weighted,
            "explanation": expl,
        })

    letter, color = _letter_grade(total)
    explanation   = _grade_explanation(pillar_scores, letter)

    return {
        "pillar_scores": pillar_scores,
        "total_score":   round(total, 1),
        "letter_grade":  letter,
        "grade_color":   color,
        "explanation":   explanation,
    }


# ── Portfolio Categorization System ──────────────────────────────────────────

_CATEGORY_META = {
    "Conservative": {
        "color":       "#10b981",
        "description": "Capital preservation focus with significant fixed-income exposure.",
    },
    "Moderate": {
        "color":       "#6366f1",
        "description": "Balanced equity/bond mix targeting steady, risk-adjusted growth.",
    },
    "Growth": {
        "color":       "#3b82f6",
        "description": "Equity-driven portfolio optimised for long-term capital appreciation.",
    },
    "Aggressive": {
        "color":       "#f59e0b",
        "description": "High-equity, high-volatility portfolio accepting significant drawdowns.",
    },
    "Resilient": {
        "color":       "#a78bfa",
        "description": "Multi-asset diversifier with inflation hedges and global reach.",
    },
}


def classify_portfolio(tickers, weights_vec, full_stats):
    """
    Classifies the portfolio into one of five categories using a scored
    rule-based approach: {Conservative, Moderate, Growth, Aggressive, Resilient}.

    Classification rules:
      Conservative → bond_pct >= 35 % OR annual vol < 10 %
      Moderate     → bond mix 15-35 % OR vol 10-16 %
      Growth       → equity >= 75 %, bond < 15 %, vol < 22 %
      Aggressive   → equity >= 90 % OR vol >= 22 % OR max single weight >= 40 %
      Resilient    → intl exposure >= 8 % AND (inflation hedge OR income tilt)
                     AND bond 5-35 %

    Returns:
      { "category": str, "color": str, "description": str }
    """
    alloc    = _classify_holdings(tickers, weights_vec)
    vol      = full_stats["vol"]
    max_w    = float(max(weights_vec))
    bond     = alloc["bond"]
    equity   = alloc["equity"]
    intl     = alloc["international"]
    infl     = alloc["inflation"]
    income   = alloc["income"]

    # Score each category. Higher score → wins.
    scores = {
        "Conservative": 0,
        "Moderate":     0,
        "Growth":       0,
        "Aggressive":   0,
        "Resilient":    0,
    }

    # ── Conservative signals
    if bond >= 0.35:    scores["Conservative"] += 40
    if bond >= 0.20:    scores["Conservative"] += 20
    if vol  < 0.10:     scores["Conservative"] += 30
    if vol  < 0.14:     scores["Conservative"] += 15

    # ── Moderate signals
    if 0.15 <= bond < 0.35:  scores["Moderate"] += 40
    if 0.10 <= vol  < 0.18:  scores["Moderate"] += 30
    if equity > 0.50:        scores["Moderate"] += 10

    # ── Growth signals
    if equity >= 0.75:  scores["Growth"] += 35
    if equity >= 0.85:  scores["Growth"] += 20
    if bond   < 0.15:   scores["Growth"] += 15
    if 0.12 <= vol < 0.22: scores["Growth"] += 20

    # ── Aggressive signals
    if equity >= 0.90:  scores["Aggressive"] += 40
    if vol    >= 0.22:  scores["Aggressive"] += 35
    if max_w  >= 0.40:  scores["Aggressive"] += 25
    if max_w  >= 0.30:  scores["Aggressive"] += 10

    # ── Resilient signals
    if intl   >= 0.08:  scores["Resilient"] += 30
    if intl   >= 0.15:  scores["Resilient"] += 15
    if infl   >= 0.03:  scores["Resilient"] += 20
    if income >= 0.05:  scores["Resilient"] += 15
    if 0.05  <= bond < 0.35: scores["Resilient"] += 20

    # Tie-break: prefer the category with the highest score
    category = max(scores, key=scores.__getitem__)
    meta     = _CATEGORY_META[category]

    return {
        "category":    category,
        "color":       meta["color"],
        "description": meta["description"],
        # expose allocation breakdown for debugging / display
        "alloc":       alloc,
        "scores":      scores,
    }
