# Twitter Sentiment Analysis

## Overview

This project classifies Twitter tweets as **positive** (0) or **negative** (1) using **Logistic Regression** with **Bag-of-Words (BoW)** and **TF-IDF** features. It preprocesses tweet text, trains a model, evaluates performance with **F1-score**, and generates test predictions for a competition (e.g., Kaggle).

## Dataset

- **Training**: 31,962 tweets (`twitter_train_data.csv`) with `id`, `label`, `tweet`.
- **Test**: Unlabeled tweets (`twitter_test_data.csv`) with `id`, `tweet`.

## Tools

- **Python**: `pandas`, `numpy`, `scikit-learn`, `nltk`, `matplotlib`, `seaborn`.
- **Jupyter Notebook**: `twitter_sentiment_final.ipynb`.
- **Git**: Version control.

## Methodology

1. **Load Data**: Read train (31,962 rows) and test datasets, combine for preprocessing.
2. **Clean Data**: Remove user mentions (`@user`) using regex, store in `tidy_tweet`.
3. **EDA**: Visualize top 10 negative hashtags (`top_negative_hashtags.png`).
4. **Feature Extraction**:
   - BoW: `CountVectorizer` (max_df=0.9, min_df=2, max_features=1000, stop_words='english').
   - TF-IDF: `TfidfVectorizer` with same parameters.
5. **Model Training**: Train Logistic Regression on 70% of training data (random_state=42).
6. **Evaluation**: Use F1-score on 30% validation set with 0.3 threshold.
7. **Prediction**: Generate test predictions, save as `sub_lreg_bow.csv`.

## KPIs

- **F1-Score (BoW)**: 0.546 (validation set).
- **F1-Score (TF-IDF)**: 0.542 (validation set).
- **Submission**: `sub_lreg_bow.csv` with test predictions.

## Project Structure

```
twitter-sentiment-analysis/
├── data/
│   ├── raw/
│   │   ├── twitter_train_data.csv
│   │   └── twitter_test_data.csv
│   ├── processed/
│   │   └── submission/
│   │       └── sub_lreg_bow.csv
├── notebooks/
│   └── twitter_sentiment_final.ipynb
├── visualizations/
│   └── top_negative_hashtags.png
├── README.md
├── requirements.txt
```

## Setup

1. Clone repo:

   ```bash
   git clone https://github.com/your-username/twitter-sentiment-analysis.git
   cd twitter-sentiment-analysis
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
   - Example `requirements.txt`:

     ```
     pandas==1.5.3
     numpy==1.24.3
     matplotlib==3.7.1
     seaborn==0.12.2
     nltk==3.8.1
     scikit-learn==1.2.2
     jupyter==1.0.0
     ```
3. Download NLTK data:

   ```python
   import nltk
   nltk.download('punkt')
   ```
4. Place datasets in `data/raw/`.
5. Run notebook:

   ```bash
   jupyter notebook notebooks/twitter_sentiment_final.ipynb
   ```

## Results

- **BoW Model**: F1-score of 0.546, slightly better than TF-IDF (0.542).
- **Hashtags**: Top negative hashtags (e.g., `#racism`) visualized.
- **Submission**: Test predictions saved in `sub_lreg_bow.csv`.

## Future Improvements

- Clean hashtags, URLs, emojis.
- Use word embeddings (e.g., BERT).
- Tune model parameters and threshold.
- Apply cross-validation or advanced models (e.g., SVM).

## Contributing

Fork, create a branch, commit changes, and open a pull request.

## License

MIT License. See LICENSE.

---

*Generated on June 19, 2025*