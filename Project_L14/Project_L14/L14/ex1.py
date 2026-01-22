import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ------------------------------------------------------------
# CpG island vs non-island using a 1st-order Markov chain
# ------------------------------------------------------------

S_ISLAND   = "ATCGATTCGATATCATACACGTAT"          # CpG+
S_NONISL   = "CTCGACTAGTATGAAGTCCACGCTTG"         # CpG-
S_QUERY    = "CAGGTTGGAAACGTAA"                   # test sequence

BASES = "ACGT"
POS   = {b: i for i, b in enumerate(BASES)}

# ------------------------------------------------------------
# Markov model building blocks
# ------------------------------------------------------------
def count_edges(seq: str):
    """Transition counts: C[i][j] = #times BASES[i] -> BASES[j]"""
    C = [[0]*4 for _ in range(4)]
    for x, y in zip(seq, seq[1:]):
        C[POS[x]][POS[y]] += 1
    return C

def normalize_rows(C):
    """Row-normalize counts into probabilities P(i->j)."""
    P = [[0.0]*4 for _ in range(4)]
    for i in range(4):
        total = sum(C[i])
        for j in range(4):
            P[i][j] = (C[i][j] / total) if total else 0.0
    return P

def llr(P_plus, P_minus, log_base=2.0):
    """
    β(i,j) = log_base( P_plus(i,j) / P_minus(i,j) )
    Zero handling: if either prob is 0 => β = 0 (handout-style).
    """
    denom = math.log(log_base)
    B = [[0.0]*4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            a, b = P_plus[i][j], P_minus[i][j]
            B[i][j] = (math.log(a / b) / denom) if (a > 0.0 and b > 0.0) else 0.0
    return B

def score(seq, B):
    """Total score is sum of β over all adjacent transitions in seq."""
    total = 0.0
    steps = []
    for k, (x, y) in enumerate(zip(seq, seq[1:]), start=1):
        v = B[POS[x]][POS[y]]
        total += v
        steps.append((k, x, y, f"{x}→{y}", v, total))
    df = pd.DataFrame(steps, columns=["step", "from", "to", "pair", "beta", "cumulative"])
    return total, df

# ------------------------------------------------------------
# Plot helpers (different colors + nicer styling)
# ------------------------------------------------------------
def heatmap(df: pd.DataFrame, title: str, cmap="viridis", center=None):
    """
    Heatmap with labels.
    - If center is provided, uses a diverging normalization around that center.
    """
    plt.figure(figsize=(5.6, 4.4))
    ax = plt.gca()

    if center is None:
        im = ax.imshow(df.values, cmap=cmap)
    else:
        vmin, vmax = float(df.values.min()), float(df.values.max())
        norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
        im = ax.imshow(df.values, cmap=cmap, norm=norm)

    ax.set_title(title)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(list(BASES)); ax.set_yticklabels(list(BASES))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # annotate cells
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{df.values[i][j]:.3f}", ha="center", va="center", fontsize=9)

    plt.tight_layout()
    plt.show()

def plot_running_score(df_steps: pd.DataFrame):
    plt.figure(figsize=(9, 4))
    ax = plt.gca()
    ax.plot(df_steps["step"], df_steps["cumulative"], marker="o", linewidth=2)
    ax.axhline(0, linestyle="--", linewidth=1)  # neutral line
    ax.set_title("Running (cumulative) log-likelihood score along the query sequence")
    ax.set_xlabel("Transition step")
    ax.set_ylabel("Cumulative score")
    ax.set_xticks(df_steps["step"])
    ax.set_xticklabels(df_steps["pair"], rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

def plot_beta_contrib(df_steps: pd.DataFrame):
    # color positives vs negatives
    df_sorted = df_steps.copy()
    df_sorted["abs_beta"] = df_sorted["beta"].abs()
    df_sorted = df_sorted.sort_values("abs_beta", ascending=False)

    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in df_sorted["beta"].values]

    plt.figure(figsize=(9, 4))
    ax = plt.gca()
    ax.bar(range(len(df_sorted)), df_sorted["beta"].values, color=colors)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title("Per-transition β contributions (sorted by |β|)")
    ax.set_xlabel("Transitions (most influential first)")
    ax.set_ylabel("β value")
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels(df_sorted["pair"].tolist(), rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# Build models from S1/S2
# ------------------------------------------------------------
C_plus  = count_edges(S_ISLAND)
C_minus = count_edges(S_NONISL)

P_plus  = normalize_rows(C_plus)
P_minus = normalize_rows(C_minus)

BETA = llr(P_plus, P_minus, log_base=2.0)

# Wrap in DataFrames for pretty printing / plotting
dfC_plus  = pd.DataFrame(C_plus,  index=list(BASES), columns=list(BASES))
dfC_minus = pd.DataFrame(C_minus, index=list(BASES), columns=list(BASES))
dfP_plus  = pd.DataFrame(P_plus,  index=list(BASES), columns=list(BASES))
dfP_minus = pd.DataFrame(P_minus, index=list(BASES), columns=list(BASES))
dfBETA    = pd.DataFrame(BETA,    index=list(BASES), columns=list(BASES))

# Score the query
final_score, df_steps = score(S_QUERY, BETA)
decision = "CpG ISLAND (+)" if final_score > 0 else "NON-ISLAND (-)"

# ------------------------------------------------------------
# Console output (all details)
# ------------------------------------------------------------
print("S1 (CpG+):", S_ISLAND)
print("S2 (CpG-):", S_NONISL)
print("S  (new): ", S_QUERY)

print("\nFinal log-likelihood score for S =", round(final_score, 6))
print("Decision:", decision)

print("\nCounts Tr+ (from S1):\n", dfC_plus)
print("\nCounts Tr- (from S2):\n", dfC_minus)

print("\nProbabilities Tr+:\n", dfP_plus.round(3))
print("\nProbabilities Tr-:\n", dfP_minus.round(3))

print("\nβ matrix = log2(Tr+/Tr-):\n", dfBETA.round(3))

print("\nPer-transition contributions:\n",
      df_steps[["step", "pair", "beta", "cumulative"]].round(6))

# ------------------------------------------------------------
# Plots (different color themes)
# ------------------------------------------------------------
heatmap(dfP_plus,  "Tr+ transition probabilities (CpG+ model)", cmap="YlGnBu")
heatmap(dfP_minus, "Tr- transition probabilities (CpG- model)", cmap="PuBuGn")

# Diverging colormap centered at 0 for β (nice for + vs - evidence)
heatmap(dfBETA, "β log-likelihood matrix: log2(Tr+/Tr-)", cmap="coolwarm", center=0.0)

plot_running_score(df_steps)
plot_beta_contrib(df_steps)
