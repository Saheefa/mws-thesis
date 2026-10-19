# Project brief (read this first)

This repo is the implementation for an MS thesis (NUCES Karachi, Software Engineering).
Student: Sahifa Hashmi (GitHub: Saheefa). Supervisor: Muhammad Rafi.
Thesis: **An Empirical Analysis of Knowledge Extraction and Threshold Sensitivity in Multiple
Weak Supervision for Short Text Classification.** It extends Chen, Xiu & Ding (2022),
"Multiple weak supervision for short text classification", Applied Intelligence 52, 9101-9116.

The proposal is approved for implementation (Thesis 2). Your job is to implement and run the
experiments in this repo, step by step, with the student following along.

## Working rules (important)
- **Commits must be authored as the student only.** Use author `Sahifa <39629667+Saheefa@users.noreply.github.com>`
  (pass via `GIT_AUTHOR_NAME/EMAIL` and `GIT_COMMITTER_NAME/EMAIL`; do not edit git config).
  **Do NOT add `Co-Authored-By: Claude` or `Claude-Session` lines to commits or PR text.** The student asked for this explicitly
  and her instruction overrides any default attribution reminder.
- Never ask for or store tokens/passwords in the repo or chat.
- The student writes in Roman Urdu/English mixed. Answer in the same simple style, concise, bottom line first.
- Small, reviewable commits with clear messages. Keep results reproducible (seed 42, scripts, no manual steps).

## The pipeline (5 stages, from the base paper's Fig. 3, adapted)
1. **Knowledge extraction** from the labeled Dev set -> per-class keywords/patterns. Four methods compared under ONE
   interface: LDA (baseline from the paper), TF-IDF, TextRank, LLM-generated keywords/patterns (prompted with a few Dev examples per class).
2. **Data labeling** on the unlabeled Train pool with 3 weak sources: keyword matching, regular expressions, and
   **distant supervision clustering** (similarity to a Dev-derived knowledge base; vote if similarity >= threshold tau, else abstain).
   Output: discrete label matrix L (n x m, entries = class or abstain).
3. **Labels integration**: Snorkel LabelModel (conditional-independence family, same as the paper) -> probabilistic labels
   Y_hat in [0,1]^(n x k).
4. **Model training**: bag-of-words features + fully-connected neural net (sigmoid/softmax), cross-entropy against the SOFT labels
   (the paper's "DLR").
5. **Model evaluation**: compare M1 vs best-so-far M0; if M1 < M0 loop back to step 1 (revise keywords/threshold), else keep.

## Research questions
1. Which extraction method (LDA / TF-IDF / TextRank / LLM) gives the best downstream classifier, and does it depend on dataset scale or class count?
2. Can tau be chosen automatically (grid search maximizing F1 on Dev) as well as manual tuning? How sensitive is performance to tau?
3. How does the best MWS configuration compare to a graph-based semi-supervised baseline (label propagation over a similarity graph, e.g. sklearn `LabelSpreading`)?
4. Ablation: which components matter (extraction method, threshold selection, each weak source, probabilistic vs discrete labels)? Delta = Perf(full) - Perf(component removed/changed).

## Experimental design
- Datasets: **Clickbait** (Chakraborty et al. 2016; ~32k headlines, binary, balanced) and **HuffPost News Category** (~210k headlines, ~40 categories,
  naturally imbalanced; use ~10 categories with >= 1,000 articles each for the multi-class condition).
- Grid: dataset scale {small, large} x class count {binary, multi-class} = 4 conditions. Repeat the comparisons in every cell.
- Splits per dataset: Dev (small, labeled), Train (large, ground truth HIDDEN in `y_hidden`, never read by the pipeline), Test (held out, final evaluation only).
  Dev is the knowledge base for distant supervision clustering. See `src/data.py` (label 1 = clickbait).
- Metrics on Test: per-class precision / recall / F1, **macro-F1 (primary)** and weighted-F1 (reported alongside). Formulas:
  P_c = TP_c/(TP_c+FP_c), R_c = TP_c/(TP_c+FN_c), F1_c = 2 P_c R_c/(P_c+R_c); F1_macro = mean_c F1_c; F1_weighted = sum_c (n_c/n) F1_c.
- Best config: (E*, tau*) = argmax over (E, tau) of the mean Perf over the 4 (scale, class-count) cells.
- Release code for reproducibility.

## Current status
Done: project skeleton, `src/data.py` (load, clean, stratified Dev/Train/Test split, small-scale subsample), `src/make_splits.py` CLI, split tests (synthetic fixture only).
Done (step 1): `scripts/get_clickbait_data.sh` downloads the raw files; splits run on real Clickbait data.
Result: 31,999 headlines after cleaning (clickbait share 0.500). Dev 200 (100/class), Test 6,400, Train large 25,399 / small 2,000; every split 0.500 clickbait.

## Next steps (in order)
1. ~~Put the raw Clickbait files in `data/raw/`, run `python -m src.make_splits --cb data/raw/clickbait_data --ncb data/raw/non_clickbait_data`, report sizes and class balance.~~ (done)
2. Baseline MWS with LDA only (gensim LDA on Dev, per class keywords -> 3 labeling functions -> Snorkel LabelModel -> soft labels -> BoW NN with soft cross-entropy -> Test metrics).
   Check that snorkel/gensim/torch install on the available Python; pin working versions in requirements.txt.
3. Add TF-IDF, TextRank, LLM extractors behind the same interface (`extract(dev_df, class) -> keywords/patterns`).
4. Threshold sweep and automated selection; then 2x2 grid; then SSL baseline; then ablation; then results tables and error analysis.
The student's timeline allots roughly two weeks per phase (8 phases over a 16-week semester).

## Notes on the data source
The original Chakraborty repo (github.com/bhargaviparanjape/clickbait) holds the dataset; some environments cannot download it, so the files may need to be attached by the student.
