# Twitter Sentiment Analysis

A machine learning project that reads a tweet and tells whether it is positive or negative.

The model is trained on 1.6 million tweets using Logistic Regression and gives 77.6% accuracy.

---

## What this project does

You give it a tweet. It tells you if the tweet is positive or negative.

That is it.

---

## Tools used

- Python
- Pandas and NumPy (for data handling)
- NLTK (for text cleaning)
- Scikit-learn (for TF-IDF and Logistic Regression)
- Google Colab (for training)
- Streamlit (for the web app)

---

## Files in this project

| File | What it is |
|------|------------|
| Twitter_Sentiment_Analysis_using_ML.ipynb | The main notebook with all the code |
| trained_model.sav | The saved model after training |
| app.py | A simple web app to test the model |
| requirements.txt | List of libraries needed to run this |

---

## How the model works

1. Take the raw tweet text
2. Remove numbers and special characters
3. Convert to lowercase
4. Remove common words like "the", "is", "a" (stopwords)
5. Reduce words to their root form — this is called stemming
6. Convert text to numbers using TF-IDF
7. Pass it through Logistic Regression
8. Get result — Positive or Negative

---

## Dataset

Name: Sentiment140
Source: Kaggle
Size: 1,600,000 tweets
Labels: 0 = Negative, 1 = Positive

---

## Model results

| | Score |
|--|--|
| Training accuracy | 79% |
| Test accuracy | 77.6% |

---

## How to run the web app

Make sure all files including trained_model.sav are in the same folder.

Then run these commands:

```
pip install -r requirements.txt
streamlit run app.py
```

It will open in your browser automatically.

---

## How to open the notebook

Open Google Colab, click on "Upload notebook", and select the .ipynb file.

---

## Author

Raushan Yadav
