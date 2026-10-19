# MWS thesis: knowledge extraction and threshold sensitivity

Implementation of *An Empirical Analysis of Knowledge Extraction and Threshold Sensitivity in
Multiple Weak Supervision for Short Text Classification* (extends Chen, Xiu & Ding, 2022).

## Pipeline
1. Knowledge extraction (LDA, TF-IDF, TextRank, LLM) from the Dev set
2. Weak labeling (keywords, regex, distant supervision clustering with threshold tau)
3. Labels integration (Snorkel label model -> probabilistic labels)
4. Soft-label classifier training (bag-of-words neural network)
5. Evaluation (precision, recall, F1, macro-F1, weighted-F1) and M1 vs M0 loop

## Layout
- `src/data.py`: loading, cleaning, Dev/Train/Test splits (label 1 = clickbait)
- `src/make_splits.py`: CLI that writes the splits to `data/processed/`
- `src/text.py`: shared tokenizer (small stop list, numbers -> NUM)
- `src/extract.py`: knowledge extraction (`extract(dev_df, cls)`; LDA so far) + shared discriminative filter
- `src/labeling.py`: 3 labeling functions (keyword, regex, distant-supervision clustering with tau)
- `src/model.py`: Snorkel LabelModel + bag-of-words NN trained on soft labels
- `src/metrics.py`, `src/run_baseline.py`: Test metrics and the end-to-end baseline run
- `results/`: JSON outputs of each run
- `tests/`: sanity checks (`python -m pytest -q tests`)
- `scripts/get_clickbait_data.sh`: downloads the raw Clickbait files into `data/raw/` (not committed)

## Data splits
Dev = 100 headlines per class (labeled); Test = 20% held out; Train = the rest with ground
truth hidden in `y_hidden` (the pipeline must never read it). Fixed seed 42. Two scales:
`large` (full Train) and `small` (2,000 Train examples).

## Run
```
pip install -r requirements.txt
bash scripts/get_clickbait_data.sh
python -m src.make_splits --cb data/raw/clickbait_data --ncb data/raw/non_clickbait_data
python -m src.run_baseline --scale large   # step 2: LDA baseline
python -m src.run_baseline --scale small
python -m pytest -q tests
```

Dataset: Chakraborty et al. (2016), *Stop Clickbait*, ASONAM. Cite it if you use the data.

## Results so far
Step 2, LDA baseline on Clickbait (tau = 0.2 manual default, seed 42; JSON in `results/`):

| scale | Train coverage | macro-F1 | weighted-F1 | accuracy | non-clickbait P / R | clickbait P / R |
|---|---|---|---|---|---|---|
| large | 0.588 | **0.759** | 0.759 | 0.769 | 0.943 / 0.572 | 0.693 / 0.966 |
| small | 0.629 | **0.789** | 0.789 | 0.795 | 0.957 / 0.619 | 0.718 / 0.972 |

Reference: the same NN trained only on the 200 labeled Dev headlines reaches macro-F1 0.862, so the
LDA baseline is below it. Diagnosis (no Train labels used): soft labels are 83% clickbait against a
50/50 Dev prior; 22 regex patterns survive for clickbait but only 1 for non-clickbait, because the
non-clickbait LDA keywords are topical (names, places) and rarely match new headlines.
