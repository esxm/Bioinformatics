"""
Plagiarism “courtroom” detector:
Eminescu (+) vs. Stănescu (-) using WORD-level Markov chains (bigrams),
log-likelihood ratio (LLR), and a sliding window scan.

Outputs:
- Two Markov models (transition probabilities between words)
- LLR scoring:
    β(w1,w2) = log2( P_E(w2|w1) / P_S(w2|w1) )
- Sliding-window classification over Mihai's text:
    score > +tau => Eminescu
    score < -tau => Stănescu
    otherwise    => Neither
- Plot like the screenshot: per-transition β curve with shaded zones (+/-)
- Optional: simulate a “Mihai” text by mixing BOTH models (Markov generator)

NOTE ABOUT COPYRIGHT:
- Eminescu is public domain.
- Stănescu is typically copyrighted. Only use excerpts you have rights to use.
"""

import re
import math
import random
from collections import Counter, defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


# =========================
# 1) Tokenization
# =========================
WORD_RE = re.compile(r"[A-Za-zĂÂÎȘȚăâîșț'-]+", re.UNICODE)

def tokenize(text: str):
    return [w.lower() for w in WORD_RE.findall(text)]


# =========================
# 2) Bigram Markov model
# =========================
class BigramModel:
    def __init__(self, tokens, alpha=1.0):
        """
        alpha: Laplace smoothing.
               Use alpha>0 for stable probabilities and LLR.
        """
        self.alpha = float(alpha)
        self.counts = defaultdict(Counter)  # counts[w1][w2]
        self.totals = Counter()             # totals[w1]
        self.vocab = set(tokens)

        for w1, w2 in zip(tokens, tokens[1:]):
            self.counts[w1][w2] += 1
            self.totals[w1] += 1
            self.vocab.add(w1)
            self.vocab.add(w2)

        self.V = max(1, len(self.vocab))

    def prob(self, w1, w2):
        """
        P(w2|w1) with Laplace smoothing.
        If w1 unseen -> uniform over vocabulary.
        """
        total = self.totals.get(w1, 0)
        if total == 0:
            return 1.0 / self.V
        c = self.counts[w1].get(w2, 0)
        return (c + self.alpha) / (total + self.alpha * self.V)

    def next_word(self, w1):
        """
        Sample a next word given w1.
        Approx sampling: observed transitions + "unseen bucket" for smoothing mass.
        """
        if self.totals.get(w1, 0) == 0:
            return random.choice(tuple(self.vocab))

        obs = self.counts[w1]
        observed_words = list(obs.keys())
        observed_weights = [(obs[w] + self.alpha) for w in observed_words]
        observed_mass = sum(observed_weights)

        unseen_count = max(0, self.V - len(observed_words))
        unseen_mass = unseen_count * self.alpha

        r = random.random() * (observed_mass + unseen_mass)

        if r < observed_mass:
            cum = 0.0
            for w, wt in zip(observed_words, observed_weights):
                cum += wt
                if r <= cum:
                    return w
            return observed_words[-1]
        else:
            return random.choice(tuple(self.vocab))


# =========================
# 3) LLR scoring
# =========================
def beta_llr(w1, w2, model_E: BigramModel, model_S: BigramModel, eps=1e-12):
    """
    β(w1,w2) = log2( P_E(w2|w1) / P_S(w2|w1) )
    """
    pe = model_E.prob(w1, w2) + eps
    ps = model_S.prob(w1, w2) + eps
    return math.log(pe / ps, 2)

def per_transition_scores(tokens, model_E, model_S):
    """
    Returns β scores for each adjacent word transition in the token list.
    Length = len(tokens)-1
    """
    return [
        beta_llr(tokens[i], tokens[i+1], model_E, model_S)
        for i in range(len(tokens) - 1)
    ]

def sliding_window_scan(tokens, model_E, model_S, win_size=30, step=10, tau=1.0):
    """
    Window score = sum of β over bigrams in the window.
    Label by tau threshold.
    """
    results = []
    n = len(tokens)
    for start in range(0, max(1, n - win_size + 1), step):
        window = tokens[start:start+win_size]
        if len(window) < 2:
            score = 0.0
        else:
            score = sum(beta_llr(a, b, model_E, model_S) for a, b in zip(window, window[1:]))

        if score > tau:
            label = "Eminescu"
        elif score < -tau:
            label = "Stanescu"
        else:
            label = "Neither"

        results.append({"start": start, "end": min(n, start+win_size), "score": score, "label": label})

    return pd.DataFrame(results)


# =========================
# 4) Simulation: “Mihai” mixed text generator
# =========================
def simulate_mihai_text(model_E, model_S, length=140, mix_p=0.5, seed_word=None, seed=0):
    """
    Generate synthetic text by mixing both Markov models:
      with probability mix_p choose Eminescu model transition
      else choose Stanescu model transition
    """
    random.seed(seed)

    vocab = list(model_E.vocab | model_S.vocab)
    if not vocab:
        return "mihai"

    if seed_word is None:
        seed_word = random.choice(vocab)

    words = [seed_word]
    for _ in range(length - 1):
        use_E = (random.random() < mix_p)
        w1 = words[-1]
        nxt = model_E.next_word(w1) if use_E else model_S.next_word(w1)
        words.append(nxt)

    return " ".join(words)


# =========================
# 5) Plot like the screenshot (per-transition β)
# =========================
def plot_plagiarism_style_transition_scores(tokens, beta_values, tau=0.25,
                                            title="Plagiarism Detection: Eminescu (+) vs. Stănescu (-)"):
    """
    tokens: list[str] words
    beta_values: list[float] of length len(tokens)-1
                 beta_values[i] is β(tokens[i] -> tokens[i+1])
    """
    if len(tokens) < 2:
        print("Not enough tokens to plot transitions.")
        return

    pairs = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
    x = list(range(len(beta_values)))
    y = beta_values

    plt.figure(figsize=(16, 8))
    ax = plt.gca()

    # Purple line with markers
    ax.plot(x, y, marker="o", linewidth=2.5, color="purple")

    # Baseline and thresholds
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axhline(+tau, color="gray", linestyle=":", linewidth=1.2, alpha=0.9)
    ax.axhline(-tau, color="gray", linestyle=":", linewidth=1.2, alpha=0.9)

    # Shaded regions
    ax.fill_between(
        x, y, 0,
        where=[v >= tau for v in y],
        interpolate=True,
        color="#4c78a8", alpha=0.25,
        label="Eminescu Zone (+)"
    )
    ax.fill_between(
        x, y, 0,
        where=[v <= -tau for v in y],
        interpolate=True,
        color="#e45756", alpha=0.25,
        label="Stănescu Zone (-)"
    )

    ax.set_title(title)
    ax.set_ylabel("Log-Likelihood Score (LLR)")
    ax.set_xlabel("Text Transitions (Scanning the Accused Text)")
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=45, ha="right")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


# =========================
# 6) Optional: visualize a small transition matrix slice (top contexts)
# =========================
def top_transitions_table(model: BigramModel, top_k=20):
    rows = []
    for w1, ctr in model.counts.items():
        for w2, c in ctr.items():
            rows.append((w1, w2, c))
    rows.sort(key=lambda t: t[2], reverse=True)
    return pd.DataFrame(rows[:top_k], columns=["from", "to", "count"])


# =========================
# 7) MAIN: paste texts and run
# =========================
if __name__ == "__main__":

    # ---- Training texts (paste your own excerpts) ----
    # Eminescu is public domain; Stănescu is often copyrighted (use text you can use).
    text_eminescu_og = """
    Vreme trece, vreme vine,
    Toate-s vechi și nouă toate;
    Ce e rău și ce e bine
    Tu te-ntreabă și socoate;
    Nu spera și nu ai teamă,
    Ce e val ca valul trece;
    De te-ndeamnă, de te cheamă,
    Tu rămâi la toate rece.

    Multe trec pe dinainte,
    În auz ne sună multe;
    Cine ține toate minte
    Și ar sta să le asculte?…
    Tu așează-te deoparte,
    Regăsindu-te pe tine,
    Când cu zgomote deșarte
    Vreme trece, vreme vine.

    Nici încline a ei limbă
    Recea cumpăn-a gândirii
    Înspre clipa ce se schimbă
    Pentru masca fericirii,
    Ce din moartea ei se naște
    Și o clipă ține poate;
    Pentru cine o cunoaște
    Toate-s vechi și nouă toate.

    Privitor ca la teatru
    Tu în lume să te-nchipui:
    Joace unul și pe patru,
    Totuși tu ghici-vei chipu-i;
    Și de plânge, de se ceartă,
    Tu în colț petreci în tine
    Și-nțelegi din a lor artă
    Ce e rău și ce e bine.

    Viitorul și trecutul
    Sunt a filei două fețe,
    Vede-n capăt începutul
    Cine știe să le-nvețe;
    Tot ce-a fost ori o să fie
    În prezent le-avem pe toate,
    Dar de-a lor zădărnicie
    Te întreabă și socoate.

    Nu spera când vezi mișeii
    La izbândă făcând punte,
    Te-or întrece nătărăii,
    De ai fi cu stea în frunte;
    Teamă n-ai, căta-vor iarăși
    Între dânșii să se plece,
    Nu te prinde lor tovarăș:
    Ce e val ca valul trece.

    Cu un cântec de sirenă,
    Lumea-ntinde lucii mreje;
    Ca să schimbe-actorii-n scenă,
    Te momesc în vârtej, rege;
    Tu pe-alături te strecoară,
    Nu băga nici chiar de seamă,
    Din cărarea ta afară
    De te-ndeamnă, de te cheamă.

    De te-ating, să feri în lături,
    De hulesc, să taci din gură;
    Ce mai vrei cu-a tale sfaturi,
    Dacă știi a lor măsură?
    Zică toți ce vor să zică,
    Treacă-n lume cine-o trece;
    Ca să nu-ndrăgești nimică,
    Tu rămâi la toate rece.
    """.strip()


    # --- Nichi ta Stănescu (copyrighted author) ---
    # I can’t paste a full Stănescu poem here, but this is an ORIGINAL,
    # sophisticated poem written to be "Stănescu-inspired" (meta, abstract),
    # ideal for building your second Markov model.
    # Nichita Stănescu — "Sunt un om viu"
# (text pasted exactly as you provided it)

    text_stanescu_og = """
    Sunt un om viu

    Sunt un om viu.
    Nimic din ce-i omenesc nu mi-e străin.
    Abia am timp să mă mir că exist, dar
    mă bucur totdeauna că sunt.

    Nu mă realizez deplin niciodată,
    pentru că
    am o idee din ce în ce mai bună
    despre viață.

    Mă cutremură diferența dintre mine
    și firul ierbii,
    dintre mine și lei,
    dintre mine și insulele de lumină
    ale stelelor.
    Dintre mine și numere,
    bunăoară între mine și 2, între mine și 3.

    Am și-un defect un păcat:
    iau în serios iarba,
    iau în serios leii,
    mișcările aproape perfecte ale cerului.
    Și-o rană întâmplătoare la mână
    mă face să văd prin ea,
    ca printr-un ochean,
    durerile lumii, războaiele.

    Dintr-o astfel de întâmplare
    mi s-a tras marea înțelegere
    pe care-o am pentru Ulise - și
    bărbatului cu chip ursuz, Dante Alighieri.

    Cu greu mi-aș putea imagina
    un pământ pustiu, rotindu-se
    în jurul soarelui...
    (Poate și fiindcă există pe lume
    astfel de versuri.)

    Îmi olace să râd, deși
    râd rar, având mereu câte o treabă,
    ori călătorind cu o plută, la nesfârșit,
    pe oceanul oval al fantaziei.

    E un spectacol de neuitat acela
    de-a ști,
    de-a descoperi
    harta universului în expansiune,
    în timp ce-ți privești
    o fotografie din copilărie!

    E un trup al tău vechi,
    pe care l-ai rătăcit
    și nici măcar un anunț, dat
    cu litere groase,
    nu-ți pferă vreo șansă
    să-l mai regăsești.

    Îmi desfac papirusul vieții
    plin de hieroglife,
    și ceea ce pot comunica
    acum, aici,
    după o descifrare anevoioasă,
    dar nu lipăsită de satisfacții,
    e un poem închinat păcii,
    ce are, pe scurt, următorul cuprins:

    Nu vreau,
    când îmi ridic tâmpla din perne,
    să se lungească-n urma mea pe paturi
    moartea,
    și-n fiece cuvânt țâșnind spre mine,
    pești putrezi să-mi arunce, ca-ntr-un râu
    oprit.

    Nici după fiecare pas,
    în golul dinapoia mea rămas,
    nu vreau
    să urce moartea-n sus, asemeni
    unei coloane de mercur,
    bolți de infern proptind deasupra-mi...

    Dar curcubeul negru-al ei, de alge,
    de-ar bate-n tinereția mea s-ar sparge.

    E o fertilitate nemaipomenită
    în pământ și-n pietre și în schelării,
    magnetic, timpul, clipită cu clipită,
    gândurile mi le-nalță
    ca pe niște trupuri vii.

    E o fertilitate nemaipomenită
    în pământ și-n pietre și în schelării.
    Umbra de mi-aș ține-o doar o clipă pironită,
    s-ar și umple de ferigi, de bălării!

    Doar chipul tău prelung iubito,
    lasă-l așa cum este, răzimat
    între două bătăi ale inimii mele,
    ca între Tigru
    și Eufrat.
    """.strip()


    # Optional: a tiny helper to normalize whitespace (useful before tokenizing)
    def normalize_text(t: str) -> str:
        t = t.replace("\t", " ")
        t = "\n".join(line.rstrip() for line in t.splitlines())
        t = "\n".join(line for line in t.splitlines() if line.strip() != "" or True)
        return t

    text_stanescu = normalize_text(text_stanescu_og)
    text_eminescu = normalize_text(text_eminescu_og)

    # ---- Accused text (Mihai) ----
    # Option A: paste your accused text here:
    text_mihai = """
Trece ziua, vine seara, și iar se închide-n mine
un ceas cu roți de liniște, pe care-l aud mergând;
mă uit la lume ca-n sală, cu masca ei care ține,
și-mi număr pașii prin gânduri, ca ploaia pe un pământ.

N-am vreme să mă mir deplin că respir în aceeași fire,
dar mă bucur — ca o scânteie — că ard, măcar o secundă;
între mine și firul ierbii stă-o prăpastie subțire,
între mine și stelele reci, o hartă care se-ntinde.

Cuvintele au umbre lungi: le rostești și ele se-ntorc,
cu gust de metal și de fruct, în cerul gurii, încet;
și uneori un singur verb îmi face sângele arc,
ridică-n aer o rană și-o schimbă-n felinar discret.

Într-o zi am înțeles că numerele nu dorm:
doi îmi stă lângă inimă, trei mi se urcă pe pleoape,
și tot ce pare întâmplare îmi scrie pe frunte un norm,
ca și cum timpul ar fi o scară ce urcă spre alte aproape.

Nu vreau, când ridic fruntea din perne, să-mi curgă-napoi
o dâră grea de întuneric, lipită de pașii mei;
vreau să rămână în urmă doar semne de vânt și de foi,
și-un râu de pace prin vorbe, să nu se oprească-n noroi.

Iar dacă lumea mă cheamă cu strigăt de miere și sare,
eu trec pe lângă chemare, cu ochiul curat și atent:
ce trece rămâne trecere, ce doare rămâne cântare,
și-n mine se face lumină, din simplul fapt că-s prezent.
""".strip()


    # ---- Parameters ----
    alpha = 1.0      # smoothing (recommended)
    tau_transition = 0.25  # threshold for per-transition shading in the screenshot-like plot
    win_size = 30
    step = 10
    tau_window = 1.0

    # 1) Build models
    tokE = tokenize(text_eminescu)
    tokS = tokenize(text_stanescu)

    model_E = BigramModel(tokE, alpha=alpha)
    model_S = BigramModel(tokS, alpha=alpha)

    # 2) Get Mihai text: use provided or simulate
    if not text_mihai.strip():
        text_mihai = simulate_mihai_text(model_E, model_S, length=120, mix_p=0.5, seed=0)
        print("\n--- Simulated “Mihai” text (mixed Markov models) ---")
        print(text_mihai)

    tokM = tokenize(text_mihai)

    # 3) Per-transition β scores + plot like screenshot
    beta_vals = per_transition_scores(tokM, model_E, model_S)

    print("\nTop transitions (Eminescu model):")
    print(top_transitions_table(model_E, top_k=12))
    print("\nTop transitions (Stanescu model):")
    print(top_transitions_table(model_S, top_k=12))

    print("\nPlotting per-transition LLR (β) with shaded zones...")
    plot_plagiarism_style_transition_scores(tokM, beta_vals, tau=tau_transition)

    # 4) Sliding-window scan (bigger stability)
    df_windows = sliding_window_scan(tokM, model_E, model_S, win_size=win_size, step=step, tau=tau_window)
    print("\nSliding-window results (first 15):")
    print(df_windows.head(15).round(3))

    # 5) Final statement (rough segmentation by window labels)
    #    (Simple: just show labeled windows; you can merge overlaps if you want.)
    print("\n--- Window attribution summary ---")
    for i, row in df_windows.iterrows():
        snippet = " ".join(tokM[int(row['start']):int(row['end'])])
        snippet = snippet[:130] + ("..." if len(snippet) > 130 else "")
        print(f"[{row['label']:8}] score={row['score']:+.3f}  tokens {int(row['start'])}-{int(row['end'])}: {snippet}")
