# Report

Put your IEEE-format report here (`.tex` source and the built `.pdf`).

Required content for Project 4:

1. **Problem & dataset**: the SMS Spam Collection, class balance, the all-ham baseline, how you handled
   duplicate messages.
2. **Text processing**: tokenization and cleaning choices, TF-IDF / counts options you tried (n-grams,
   `min_df`, sublinear tf) and why. State that the vocabulary and IDF were fitted on training data only.
3. **Naive Bayes (yours)**: smoothing, log space, agreement with scikit-learn
   (`reference_check` in `results/metrics.json`), and the effect of alpha including alpha = 0
   (`results/alpha_sweep_*.png`).
4. **Other classifiers**: linear SVM (how you chose C, using training data only), KNN with k = 1, 3, 5, and
   the exploratory k-means clustering (what the clusters look like vs. spam/ham).
5. **Evaluation**: both protocols (`paper` and `stratified`); precision, recall, F1, MCC, Spam Caught,
   Blocked Ham, ROC-AUC / PR-AUC; confusion matrices. Justify your operating point: what does a blocked
   legitimate message cost compared with a missed spam?
6. **Comparison with the paper** (Almeida et al., 2011): where your numbers agree or differ, and why
   (duplicates, protocol, preprocessing, calibration of Naive Bayes probabilities).
7. **Conclusion & limitations.**

Every number in the report must match `results/metrics.json`.
