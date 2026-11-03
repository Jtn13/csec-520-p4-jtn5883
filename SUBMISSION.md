# Submission Manifest: Project 4

Fill this in before submitting. The grading agent reads it first.

## Student
- Name (RIT ID): Jack Nardi (jtn5883)

## Course level
- [x] 520 (undergraduate)
- [ ] 620 (graduate)

## Dataset
- Name / source / version / retrieval date: SMS Spam Collection v.1 (Almeida, Gómez Hidalgo & Yamakami, 2011), UCI Machine Learning Repository, dataset 228 (https://archive.ics.uci.edu/dataset/228/sms+spam+collection). Retrieved [DATE]. 5,574 messages: 4,827 ham, 747 spam.
- How to obtain it (command or steps): from the repo root:
  ```bash
  curl -L -o data/smsspamcollection.zip "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
  unzip data/smsspamcollection.zip -d data/
  ```
  This gives `data/SMSSpamCollection`, which `config.yaml` reads (`data.source: file`). The corpus is git-ignored. Details are in `data/README.md`.
- Duplicate messages: removed? how many? what effect on your results?
  - Yes. The main run (`config.yaml`, `data.deduplicate: true`) removes 403 exact duplicates before splitting, leaving 5,171 messages (4,518 ham, 653 spam). Spam is duplicated at about twice the rate of ham (94 of 747 spam, 12.6%, vs 309 of 4,827 ham, 6.4%), which fits templated spam campaigns.
  - Without deduplication, 206 of the 3,902 paper-split test messages (208 stratified) also appear in the training set.
  - Effect, from `results/nodedup/metrics.json` (identical config except `deduplicate: false`): on the paper split, keeping duplicates raises MCC by about 0.01 to 0.02 for every model. On the stratified split, KNN still gains (+0.017 to +0.044 MCC) while Naive Bayes and the SVM do not (-0.003 and -0.007). KNN benefits most because a test message with an exact copy in training is matched directly. The two runs have different test sets, and these are single-seed results, so the small NB and SVM differences are within noise.

## How to reproduce
```bash
make setup
make reproduce
```
Anything non-default the grader must know (data download, runtime):
- Download the data first (see Dataset above).
- `make reproduce` runs the main config (`config.yaml` -> `results/`). The duplicate comparison is a second config: `python -m src.run --config config_nodedup.yaml` -> `results/nodedup/`.
- Runtime: [RUNTIME]. The SVM grid search and the out-of-fold threshold selection add a few extra fits per protocol.

## Text processing
- Features used in the reported run (`config.yaml -> features`): kind, n-grams, min_df, sublinear_tf:
  - `kind: tfidf` for the SVM, KNN and k-means; `nb_kind: counts` for Naive Bayes (separate vectorizer, same settings, also fit on training text only).
  - Unigrams (`ngram_range: [1, 1]`), `min_df: 1`, `sublinear_tf: false`, `lowercase: true`, no `max_features` cap.
- Preprocessing choices and why:
  - Lowercasing only. No stopword removal and no stemming, matching the paper, which cites work showing both tend to hurt spam filtering.
  - scikit-learn's default token pattern, which keeps words of two or more letters or digits. It drops single-character tokens ("u", "2") and symbols like "£" and "!", so it differs from the paper's tok1 and tok2 tokenizers.
  - Counts for NB, TF-IDF for everything else, chosen from an ablation on the no-dedup data: NB on counts reached MCC 0.918 vs 0.703 on TF-IDF (paper split), while the SVM and KNN did better on TF-IDF (SVM 0.920 vs 0.897). Multinomial NB models word counts, and on L2-normalized TF-IDF rows the alpha = 1 smoothing swamps the evidence, so NB falls back on the ham prior (precision 1.0, only about 53% of spam caught).
- Confirm the vectorizer is fitted on training text only: [x] Also inside cross-validation: the SVM grid search and the threshold selection put the vectorizer inside each fold, so validation folds never contribute to the vocabulary or IDF weights.

## Your Naive Bayes
- Confirm `config.yaml` has `naive_bayes.implementation: scratch`: [x]
- Smoothing alpha used, and why: alpha = 1.0 (standard Laplace smoothing). It was set in advance, not picked from the alpha sweep, because that sweep is scored on the test set. The sweep shows alpha = 0 collapses spam caught to about 49% (paper split) with non-finite log-probabilities: any word never seen in a class gets log P = -inf, so a single such word vetoes that class. Values from 0.01 to 1 perform similarly, and larger alpha trades spam caught for fewer blocked ham.
- Agreement with the reference (`reference_check` in metrics.json):
  - `prediction_agreement`: 1.0 (both protocols)
  - `max_abs_diff_log_prob`: 0.0 (both protocols; `max_abs_diff_feature_log_prob` is also 0.0)

## Other classifiers
- Linear SVM: how C was chosen (training data only): `GridSearchCV` over C in {0.01, 0.1, 1, 10, 100} with 5-fold stratified CV on the training messages, scored by spam-class F1, with the TF-IDF vectorizer inside the pipeline. C = 10 was chosen on both protocols (CV F1 0.946 paper, 0.914 stratified). C = 10 and C = 100 tied exactly, and the smaller value was kept. C = 0.01 predicted everything as ham (F1 = 0).
- KNN: k values and metric: k = 1, 3, 5 (as in the paper), cosine distance on TF-IDF, brute-force search.
- k-means exploration: what it showed: my Project 2 `KMeansScratch` (`kmeans.implementation: p2`) on 50 LSI components of the training TF-IDF vectors.
  - At k = 2 the clusters do not follow spam vs ham (V-measure 0.014 paper, 0.012 stratified).
  - At k = 10 (reported config) one cluster is 75% spam (paper) / 87% spam (stratified), while most others are 0 to 5% spam (V-measure 0.110 / 0.207).
  - Homogeneity is much higher than completeness (stratified 0.56 vs 0.13): spam templates group into one fairly pure cluster, but ham is varied and spreads across many clusters, so the dominant structure in the text is not spam vs ham.
  - k = 5 was also explored and was unstable (spam cluster found on the stratified split only).

## Claimed results (must match `results/metrics.json`, protocol: paper, first 30% train / 70% test, deduplicated)
| Classifier | Precision | Recall (SC%) | F1 | MCC | Blocked ham % |
|---|---|---|---|---|---|
| Naive Bayes (yours) | 0.9789 | 85.68 | 0.9138 | 0.9055 | 0.25 |
| Linear SVM | 0.9814 | 85.45 | 0.9136 | 0.9055 | 0.22 |
| KNN (best k = 1, by MCC) | 0.9485 | 80.83 | 0.8728 | 0.8606 | 0.60 |
| Paper's best (SVM) | not reported | 83.10 | not reported | 0.893 | 0.18 |

the paper evaluated without removing duplicates. The like-for-like comparison is `results/nodedup/metrics.json`, paper split: my SVM reaches SC 87.65%, BH 0.27%, MCC 0.917 and my NB reaches SC 88.43%, BH 0.35%, MCC 0.918. Stratified results for the main run are in `results/metrics.json` (SVM MCC 0.925, NB 0.910).

- Operating point / threshold and the cost argument behind it:
  - A blocked legitimate message costs more than a missed spam, so the threshold is the one that catches the most spam while blocking at most 0.5% of ham (`operating_point.max_blocked_ham_pct: 0.5`). The 0.5% cap is in the range of the paper's best filters (0.18% to 0.53% blocked ham).
  - The threshold is chosen on 5-fold out-of-fold scores from the training set only, then applied to the test set once. Ties go to the higher (more conservative) threshold. Results are the `naive_bayes_op` and `linear_svm_op` entries.
  - The chosen thresholds landed close to the defaults (paper split: NB spam log-odds 0.207, SVM decision value -0.006, both defaults 0), so the default operating points were already near-optimal under this cost constraint. NB at its operating point blocked one fewer legitimate message (BH 0.22%, MCC 0.9069). Test blocked ham stayed under the cap for both models on both protocols.
  - In the no-dedup stratified run, NB's training-chosen threshold (-0.92) did not transfer: test blocked ham was 1.12%, more than twice the cap, while the SVM's held (0.44%). NB's log-odds are extreme and poorly calibrated, so thresholds tuned on its training scores are unstable.

## Exploratory task (Enron-Spam), if done (bonus: up to 10 extra points)
- Which Enron set(s) and what you did (pipeline on Enron / cross-corpus): Not done.
- Main finding (transfer SMS <-> Enron, differences from the SMS results): N/A.

## AI-use acknowledgment
Per the syllabus policy, briefly note any substantive use of AI assistants.

I used Claude as an aid throughout this project:
- Planning the project steps and explaining the multinomial Naive Bayes math (priors, add-alpha smoothing, log-space scoring, log-sum-exp). I wrote `NaiveBayesScratch` myself. Claude reviewed my code, pointed out a syntax error, and tested it against scikit-learn.
- Help interpreting results (duplicates, features ablation, k-means, comparison with the paper) and creating tables for my report from the data