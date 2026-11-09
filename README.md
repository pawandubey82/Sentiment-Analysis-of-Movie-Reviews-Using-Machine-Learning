# 🎬 Sentiment Analysis of Movie Reviews Using Machine Learning

## 📌 Project Overview

This project focuses on analyzing movie reviews and classifying them into two sentiment categories: **Positive** and **Negative** using Machine Learning techniques.

The project uses the IMDb Movie Reviews dataset containing 50,000 movie reviews. Natural Language Processing (NLP) and TF-IDF vectorization are used to convert text reviews into numerical features for machine learning models.

## 🎯 Objectives

- Analyze movie reviews to understand customer opinions.
- Perform data cleaning and preprocessing.
- Explore positive and negative sentiment patterns using EDA.
- Apply Natural Language Processing techniques.
- Train and compare multiple Machine Learning models.
- Evaluate model performance using classification metrics.
- Identify the best-performing model for sentiment prediction.

## 📂 Dataset Information

- **Dataset:** IMDb Dataset of 50K Movie Reviews
- **Total Records:** 50000
- **Features:** 2
- **Input Feature:** Review
- **Target Variable:** Sentiment
- **Classes:** Positive and Negative

**Dataset Source:**  
[Kaggle – IMDb Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

The dataset contains movie reviews labelled as positive or negative.

## 🛠️ Technologies Used

- Python
- Google Colab
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- NLP
- TF-IDF Vectorization

## ⚙️ Project Workflow

### 1. Data Cleaning and Preprocessing

- Handling missing values.
- Removing duplicate reviews.
- Cleaning HTML tags and unnecessary spaces.
- Converting sentiment labels into numerical values.
- Preparing text data for machine learning.

### 2. Exploratory Data Analysis (EDA)

- Positive and negative sentiment distribution.
- Review length analysis.
- Word frequency analysis.
- Correlation heatmap.
- Statistical analysis using Mann-Whitney U test and Chi-square test.

### 3. Feature Extraction

TF-IDF (Term Frequency–Inverse Document Frequency) is used to convert text reviews into numerical feature vectors.

### 4. Machine Learning Models

Three machine learning algorithms are implemented:

1. Multinomial Naive Bayes
2. Logistic Regression
3. Linear Support Vector Machine (SVM)

GridSearchCV with 3-fold cross-validation is used for hyperparameter tuning.

### 5. Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- ROC Curve

## 📊 Model Performance

Best Cross-Validation F1 Scores obtained during hyperparameter tuning:

| Model | Best CV F1 Score |
|---|---:|
| Naive Bayes | 87.63% |
| Logistic Regression | 89.89% |
| Linear SVM | 90.04% |

**Best Cross-Validation Model:** Linear SVM

*Note: These are cross-validation F1 scores, not final test-set scores.*

## 📁 Project Structure

```text
Sentiment-Analysis-of-Movie-Reviews-Using-Machine-Learning/
│
├── README.md
├── sentiment_analysis.ipynb
├── requirements.txt
├── model_comparison.csv
├── test_predictions.csv
└── best_sentiment_model.pkl
```

*Output files are included when generated and saved.*

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/pawandubey82/Sentiment-Analysis-of-Movie-Reviews-Using-Machine-Learning.git
```

### 2. Open the project folder

```bash
cd Sentiment-Analysis-of-Movie-Reviews-Using-Machine-Learning
```

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy joblib
```

### 4. Download the IMDb dataset

Download the dataset from Kaggle and place `IMDB Dataset.csv` in the project directory.

### 5. Open Jupyter Notebook

```bash
jupyter notebook
```

### 6. Run the notebook

Open `sentiment_analysis.ipynb` and execute all cells sequentially.

## 💡 Key Learnings

- Understanding Natural Language Processing.
- Working with real-world text datasets.
- Performing exploratory data analysis.
- Implementing machine learning classification algorithms.
- Hyperparameter tuning using GridSearchCV.
- Evaluating and comparing machine learning models.

## ⚠️ Limitations

- The model is trained on IMDb movie reviews and may not generalize to every type of text.
- Sarcasm and complex expressions can be difficult to classify.
- The model uses traditional TF-IDF features rather than advanced contextual language models.

## 🔮 Future Improvements

- Implement deep learning models such as LSTM and Bi-LSTM.
- Experiment with BERT-based sentiment classification.
- Develop a real-time sentiment analysis web application.
- Improve handling of sarcasm and complex language.

## 👨‍💻 Author

**Pawan Dubey**

- **GitHub:** [@pawandubey82](https://github.com/pawandubey82)
- **Repository:** [Sentiment Analysis of Movie Reviews Using Machine Learning](https://github.com/pawandubey82/Sentiment-Analysis-of-Movie-Reviews-Using-Machine-Learning)

**Skills:** Machine Learning | Data Science | Python | NLP

---

⭐ If you find this project useful, consider giving this repository a star!
