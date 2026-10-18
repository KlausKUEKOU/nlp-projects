# NLP Projects

Natural Language Processing work in Python: tokenization, text classification, sentiment analysis
with Hugging Face models, and unsupervised clustering of short texts.

| Notebook | Question | Methods |
|---|---|---|
| [`01_tokenization_benchmark`](notebooks/01_tokenization_benchmark.ipynb) | Does the tokenization strategy change sentiment classification accuracy on Amazon reviews? | Character, word, whitespace, sentence and BERT sub-word tokenization, bag-of-words classifier |
| [`02_text_classification_tests`](notebooks/02_text_classification_tests.ipynb) | Classification experiments supporting the benchmark | scikit-learn pipelines |
| [`03_survey_responses_clustering`](notebooks/03_survey_responses_clustering.ipynb) | What themes come out of open answers about remote work? (French) | spaCy preprocessing, TF-IDF, K-Means, word clouds |
| [`04_whatsapp_forum_analysis`](notebooks/04_whatsapp_forum_analysis.ipynb) | What does a 2,772-message student WhatsApp group talk about, and is it friendly? | Parsing and cleaning chat exports, emoji handling, word frequencies, multilingual BERT sentiment (1 to 5 stars) |

## Tokenization benchmark

Amazon Fine Food Reviews (first 10,000 reviews, 3 sentiment classes):

| Tokenization | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Characters | 0.8125 | 0.8016 | 0.8125 | 0.8061 |
| Words | 0.8110 | 0.8003 | 0.8110 | 0.8040 |
| Sub-words (BERT) | **0.8145** | 0.7956 | **0.8145** | 0.8015 |
| Whitespace | 0.8105 | **0.8039** | 0.8105 | **0.8066** |
| Sentences | 0.7670 | 0.7540 | 0.7670 | 0.6768 |

With sentence tokenization the model predicts mostly the majority class. The four other methods
are within one point of each other. Confusion matrices: [`results/results.txt`](results/results.txt).
Full rendered report: [`reports/tokenization_benchmark_report.html`](reports/tokenization_benchmark_report.html).

## Run it

```bash
pip install pandas numpy scikit-learn nltk spacy transformers torch matplotlib seaborn wordcloud emoji unidecode
```

- Notebook 01: download `Reviews.csv` from
  [Kaggle: Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) into `data/`.
- Notebook 04 expects a WhatsApp chat export (`.txt`, "Export chat" without media); it is not included
  to protect participants' privacy.

Each notebook also opens in Google Colab from the badge at its top.

**Author:** Klaus KUEKOU
