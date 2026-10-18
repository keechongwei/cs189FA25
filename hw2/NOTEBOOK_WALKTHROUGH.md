# HW2 — Notebook Walkthrough (LMArena Leaderboards & Style Control)

Notebooks covered: `arena_warmup.ipynb`, `arena_style_control.ipynb` (plus helper module `plotting_utils.py`)

**Goal in the top-level README:** "neural controllers for an interactive environment", policy training under different objectives, and "style conditioning" to control behaviour.

> **Mismatch with the README.** These notebooks do not build controllers or policies. Homework 2 is about **ranking large language models from human preference data** (LMArena / Chatbot Arena) using logistic regression, bootstrapping and stylistic confounders. The word "style" refers to the *style of an LLM's responses* (length, markdown formatting), and "control" refers to *statistically controlling for those features* in a regression. The sections below describe what the code actually does. The README description for HW2 is worth updating.

---

## Libraries used

| Library | Used for |
|---|---|
| `datasets` (Hugging Face) | `load_dataset("lmarena-ai/arena-human-preference-100k")` (106,134 battles) |
| `pandas`, `numpy` | battle tables, feature matrices, bootstrapping |
| `plotly` (`express`, `graph_objects`) | heatmaps, bar charts, error-bar plots (`pd.options.plotting.backend = "plotly"`) |
| `scikit-learn` | `LogisticRegression`, `TfidfVectorizer`, `KMeans`, `TruncatedSVD` |
| `tiktoken` | GPT-2 token counts of responses |
| `gradio` | interactive viewer of prompt clusters |
| `huggingface_hub`, `otter-grader` | login and autograder |
| `plotting_utils.py` (local) | `plot_rank_heatmap`, `plot_style_features` |

---

## `arena_warmup.ipynb` — exploring the data and building a naive leaderboard

### Q1: Exploratory analysis
- Each row is one battle: `model_a`, `model_b`, `winner` (`model_a`, `model_b`, `tie`, `tie (bothbad)`), two conversations, language, turn count and category tags.
- **1a:** top-20 models by combined `model_a`+`model_b` appearances (`pd.concat(...).value_counts()[:20]`).
- **1b:** `subselect_battles` keeps battles where *both* models are in the top 20 (`isin`), plus a second version with ties removed (about 26k of about 40k).
- A pivot-table heatmap (`ptbl + ptbl.T`) shows how often each pair meets.
- **1c (written):** strong-vs-strong pairs dominate because LMArena uses weighted sampling to separate close models, which needs more data. Matching strong models against weak ones also inflates win rates.
- Plots of language and turn-count distributions follow.

### Q2: Average win rate leaderboard
- `compute_pairwise_win_fraction` builds a 20×20 matrix where cell (i, j) is the fraction of i-vs-j battles won by i. Missing pairs and the diagonal are `NaN`. The matrix is sorted by each row's mean.
- The **average win rate (AWR)** is the row mean, shown as a ranked bar chart.
- **2c (written):** AWR can misrepresent strength. It ignores that models face different opponents and a different number of battles. A model that mostly beat weak opponents ranks too high, and rarely-paired matchups are noisy.

### Q3: Prompt analysis
- **First user message** is extracted from `conversation_a`; English (plus `unknown`) battles between top-20 models are kept.
- **3a:** removing the 10 most common prompts (mostly "hi", "hello", prompt-injection text and trivial arithmetic) reorders the top 5. The GPT-4o family drops (e.g. `gpt-4o-2024-05-13` from 3rd to 5th). The written explanation: these models may do comparatively well on greetings and short chat.
- **3b:** boolean category tags (`creativity`, `technical_accuracy`, `if`, `math`, `is_code`) give a bar chart of category proportions. A per-category AWR is stacked into a tidy DataFrame (`model`, `category`, `win rate`, `rank`) and drawn with `plot_category_rank_heatmap`. Rankings shift by category.
- **3c: K-means on prompt text.** 8,000 sampled prompts → `TfidfVectorizer(max_features=500, min_df=5)` → `KMeans` for K ∈ {4, 6, 8, 10, 12}. Inertia and runtime are plotted against K, and **K = 8** is chosen from the elbow. Cluster sizes: 2702, 1527, 1388, 640, 636, 398, 357, 352.
  - A 2-D `TruncatedSVD` projection coloured by cluster is built into a Gradio app.
  - **3d (written):** cluster 5 is mostly "write …" creative-writing prompts; the largest cluster (0) is a mixed bag, typical of TF-IDF + K-means.
- **3e: engineered leaderboard shift.** `filter_out_battles` drops prompts containing `write|Write|story|Story`, which removes about 14% (8000 → 6872). The top-5 ranking changes, supporting the hypothesis that the top models are comparatively strong at creative prompts.

---

## `arena_style_control.ipynb` — Bradley-Terry strengths, confidence intervals, style control

The warm-up's data selection is repeated. Battles are also **deduplicated** using `dedup_tag["sampled"]`, leaving 25,533 battles.

### Q4: Bradley–Terry model via logistic regression
- `turn_into_features` makes **two rows per battle**: model A = +1 / model B = −1 with label "A won", and the mirrored row. This gives X of shape **(51,066 × 20)** and a label vector y.
- `LogisticRegression(fit_intercept=False)`; the coefficients are the strength scores *S_m*, and P(A beats B) = σ(S_A − S_B).
- **4c (written):** no intercept is needed because the data is symmetric. An intercept would add the same constant to every score and cancel in S_A − S_B.
- The top scores are `chatgpt-4o-latest` (≈ 0.76), `gemini-1.5-pro-exp-0801` (≈ 0.57) and `gpt-4o-2024-05-13` (≈ 0.42).

### Q5: Bootstrapped confidence intervals
- `get_bootstrapped_score` resamples battles with replacement, refits the regression each time, and reports the mean score with the 2.5th and 97.5th percentiles.
- **Rank** = 1 + the number of models that are *confidently* better (their lower bound exceeds this model's upper bound). So overlapping models share a rank.
- The notebook uses `n_bootstrap=10` for speed. That is a small number for stable 95% intervals, but it is the default in the assignment's call.
- `plot_rank_heatmap` visualises the rankings.

### Q7: Does response length influence rank?
- `calculate_response_length` concatenates assistant messages with `"\n\n"` and counts GPT-2 tokens with `tiktoken` (`disallowed_special=()`).
- `per_model_battles` reshapes to one row per (model, battle). Average length per model, ordered by rank, is plotted with a `np.polyfit` trendline. Higher-ranked models tend to write longer answers.

### Q8: Style control
- **8a `add_style_features`:** reads `conv_metadata` and computes a normalised difference `(a − b) / (a + b)` (0 if both are 0) for bold count, header count, list count and total assistant tokens.
- **8b `make_pairwise_feature_df`:** builds a 2-rows-per-battle table (A→B and B→A) with the model indicator vector, the label, and sign-flipped style features. (A latent quirk: `y` here compares `winner` with the model *name*, so its labels are wrong. It is not used downstream, because the pipeline reuses the `y` from Q4.)
- **8c `get_sc_category_results`:** `np.hstack([X, X_style])` is fed through the same bootstrapped logistic regression. Model coefficients are then the strengths *controlling for style*; style coefficients are ranked −1.
- **Results (from the notebook's commentary and plots):**
  - **Length is the strongest feature**, with a coefficient larger than any model's score.
  - Bold and list effects are smaller.
  - After control, `gpt-4o-mini` and `llama-3.1-70b-instruct` drop, and `claude-3-5-sonnet` rises.

### Q9: Discovering new style features
- `tfidf_phrase_diff` fits a 2–4-gram TF-IDF on **winning vs losing** assistant text and lists the phrases with the largest difference in mean TF-IDF.
- Apologetic phrases ("sorry", "I apologize") show up more in losers, which motivates the `refusal_count` feature (binary A-vs-B difference via `count_phrase_diff`).
- **9b:** own feature `exclamation_marks`, the per-word rate of "!" in A minus B. Both features are added to the style list and the leaderboard and coefficient plots are regenerated.
- **9c (written):** earlier attempts (enthusiasm phrases like "Great!", certainty phrases like "Absolutely!") produced little change in rankings. The chosen exclamation-mark feature is the final one kept.
