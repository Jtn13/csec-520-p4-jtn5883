# Data

**Do not commit datasets.** This directory is git-ignored (except this file).

## Default: the bundled sample (no download)

`config.yaml` ships with `data.source: sample`, a 58-line toy corpus in
`tests/fixtures/sample_sms.tsv`. It is only for debugging your Naive Bayes quickly: the
numbers it produces mean nothing.

## Project dataset: SMS Spam Collection

1. Download the **SMS Spam Collection v.1** (Almeida, Gomez Hidalgo & Yamakami), from the UCI
   Machine Learning Repository: <https://archive.ics.uci.edu/dataset/228/sms+spam+collection>.
   The course copy (`smsspamcollection.zip`) is on the course drive.
2. Unzip it and place the file named `SMSSpamCollection` in this directory (it stays untracked).
3. Point `config.yaml` at it:

```yaml
data:
  source: file
  path: data/SMSSpamCollection
```

Format: one message per line, `label<TAB>text`, with label `ham` or `spam`.
**5,574 messages: 4,827 ham (86.6%) and 747 spam (13.4%).**

### Things to know (and to report on)

- **Duplicates.** The corpus contains about 400 extra copies of repeated messages. In the paper's
  split (first 30% for training), roughly 200 test messages also occur in the training set.
  `data.deduplicate: true` (the default) removes exact duplicates before splitting. Say in your
  report what you chose and what it did to your numbers.
- **File order is not chronological**, so the "paper" protocol is not a time split.
- **Baseline to beat:** a filter that labels everything ham has 86.6% accuracy and catches no spam.
- Record the source, version and retrieval date of your copy here when you submit.

### Citation

Almeida, T. A., Gomez Hidalgo, J. M., & Yamakami, A. (2011). *Contributions to the Study of SMS
Spam Filtering: New Collection and Results.* Proceedings of the 2011 ACM Symposium on Document
Engineering (DocEng'11).

## Exploratory dataset: Enron-Spam (larger, e-mail)

A larger, labeled e-mail corpus from the same literature (**Metsis, Androutsopoulos & Paliouras, 2006**):
six datasets built from the legitimate mail of Enron employees plus spam, about 33,000 messages in total.

1. Go to <https://www2.aueb.gr/users/ion/data/enron-spam/> (readme: <http://www.aueb.gr/users/ion/data/enron-spam/readme.txt>).
2. Download the **preprocessed** sets, for example
   `http://www.aueb.gr/users/ion/data/enron-spam/preprocessed/enron1.tar.gz` (about 1.8 MB; `enron2` ... `enron6` are the others).
   `raw/` has the original messages (duplicates kept) if you prefer to do your own preprocessing.
3. Extract into `data/` so you have `data/enron1/ham/` and `data/enron1/spam/`. These stay untracked.
4. Use it in `config.yaml`:

```yaml
data:
  enron_path: data/enron1
evaluate:
  cross_corpus: true      # train on SMS, test on Enron, and the reverse (written to metrics.json -> cross_corpus)
```

   or run the entire pipeline on Enron instead of SMS with `data.source: enron`.

Notes: each message is its own text file; the number and date in the file name give the arrival order, which
the loader uses so that the `paper` protocol is chronological for Enron. Messages are long, with a subject line
and sometimes HTML, unlike 160-character SMS, so preprocessing matters more. Class balance also differs from
the SMS set (about half spam), so the all-ham baseline and the base-rate discussion change. Record the
file(s) and retrieval date in your report.

Citation: V. Metsis, I. Androutsopoulos and G. Paliouras, *Spam Filtering with Naive Bayes - Which Naive
Bayes?* Proceedings of the 3rd Conference on Email and Anti-Spam (CEAS 2006), Mountain View, CA.

