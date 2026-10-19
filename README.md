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
- `tests/`: sanity checks (`python tests/test_splits.py`)
- `data/raw/`: put the original dataset files here (not committed)

## Data splits
Dev = 100 headlines per class (labeled); Test = 20% held out; Train = the rest with ground
truth hidden in `y_hidden` (the pipeline must never read it). Fixed seed 42. Two scales:
`large` (full Train) and `small` (2,000 Train examples).

## Run
```
pip install -r requirements.txt
python -m src.make_splits --cb data/raw/clickbait_data --ncb data/raw/non_clickbait_data
python tests/test_splits.py
```

Dataset: Chakraborty et al. (2016), *Stop Clickbait*, ASONAM. Cite it if you use the data.
