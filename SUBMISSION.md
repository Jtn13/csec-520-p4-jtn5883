# Submission Manifest: Project 4

Fill this in before submitting. The grading agent reads it first.

## Student
- Name (RIT ID):

## Course level
- [ ] 520 (undergraduate)
- [ ] 620 (graduate)

## Dataset
- Name / source / version / retrieval date:
- How to obtain it (command or steps):
- Duplicate messages: removed? how many? what effect on your results?

## How to reproduce
```bash
make setup
make reproduce
```
Anything non-default the grader must know (data download, runtime):

## Text processing
- Features used in the reported run (`config.yaml -> features`): kind, n-grams, min_df, sublinear_tf:
- Preprocessing choices and why:
- Confirm the vectorizer is fitted on training text only: [ ]

## Your Naive Bayes
- Confirm `config.yaml` has `naive_bayes.implementation: scratch`: [ ]
- Smoothing alpha used, and why:
- Agreement with the reference (`reference_check` in metrics.json):
  - `prediction_agreement`:
  - `max_abs_diff_log_prob`:

## Other classifiers
- Linear SVM: how C was chosen (training data only):
- KNN: k values and metric:
- k-means exploration: what it showed:

## Claimed results (must match `results/metrics.json`, protocol: ______)
| Classifier | Precision | Recall (SC%) | F1 | MCC | Blocked ham % |
|---|---|---|---|---|---|
| Naive Bayes (yours) | | | | | |
| Linear SVM | | | | | |
| KNN (best k) | | | | | |
| Paper's best (SVM) | | | | | |

- Operating point / threshold and the cost argument behind it:

## Exploratory task (Enron-Spam), if done (bonus: up to 10 extra points)
- Which Enron set(s) and what you did (pipeline on Enron / cross-corpus):
- Main finding (transfer SMS <-> Enron, differences from the SMS results):

## AI-use acknowledgment
Per the syllabus policy, briefly note any substantive use of AI assistants.
