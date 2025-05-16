# 📰 Fake News Detection Using Machine Learning


![YouTube Banner - Fake News Detection Project Overview](https://github.com/user-attachments/assets/78e11973-fc9c-47ff-a962-998bbcce9813)

---

## 📛 Problem Statement

In the digital age, misinformation and fake news can spread rapidly across social media and online platforms. Such content may influence public perception, affect elections, incite panic, or sway markets. To preserve the integrity of information, there is a growing need for automated systems capable of verifying the authenticity of online news articles.

---

## 🌟 Objective

* To build a robust Machine Learning model to classify news articles as **Real** or **Fake**.
* To leverage **Natural Language Processing (NLP)** techniques for feature engineering from raw text.
* To evaluate and compare multiple ML models using cross-validation.
* To implement a predictive system that classifies unseen news inputs.

---

## 📚 Dataset Overview

| Column   | Description                              |
| -------- | ---------------------------------------- |
| `id`     | Unique identifier for each news article  |
| `title`  | Title of the news article                |
| `author` | Author of the news article               |
| `text`   | Main body of the news article            |
| `label`  | Target: `0` = Real News, `1` = Fake News |

* Dataset Size: **20,800 rows × 5 columns**

---

## 🛠️ Technologies and Libraries Used

* **Python 3.8+**
* **Pandas**, **NumPy** – Data manipulation
* **NLTK** – Stopwords removal, stemming
* **Regular Expressions** – Text cleaning
* **Scikit-learn**:

  * `TfidfVectorizer` – Text to feature matrix
  * `train_test_split`, `cross_val_score`, `StratifiedKFold`
  * **Models**: LogisticRegression, PassiveAggressiveClassifier, MultinomialNB, LinearSVC, RandomForestClassifier
  * Model evaluation: accuracy, precision, recall, F1 score

---

## 🛠️ Project Workflow

### 1. Data Preprocessing

* Loaded the dataset and replaced missing values with empty strings
* Merged `author` and `title` into a new column `content`
* Applied:

  * Lowercasing
  * Removal of non-alphabetic characters
  * Stopword removal using NLTK
  * Stemming using **PorterStemmer**

### 2. Feature Extraction

* Used **TfidfVectorizer** to convert the cleaned content into numerical features
* Resulting feature matrix: **(20800, 17128)** (sparse format)

### 3. Model Building

* Split the data into train (80%) and test (20%) sets using **stratified sampling**
* Implemented and trained **five models**:

  * Logistic Regression
  * Passive Aggressive Classifier
  * Multinomial Naive Bayes
  * Linear SVM (LinearSVC)
  * Random Forest Classifier

### 4. Model Evaluation (5-Fold Cross-Validation)

| Model                   | Accuracy (mean ± std) |
| ----------------------- | --------------------- |
| Logistic Regression     | 98.23% ± 0.17%        |
| Passive Aggressive      | 97.94% ± 0.22%        |
| Multinomial Naive Bayes | 96.62% ± 0.19%        |
| Linear SVM              | **98.51% ± 0.15%**    |
| Random Forest           | 97.27% ± 0.21%        |

* **Best Model**: LinearSVC (Support Vector Machine)

### 5. Final Evaluation on Test Set

* Trained the best model on the training set
* Evaluated on the test set

| Metric    | Value      |
| --------- | ---------- |
| Accuracy  | **98.41%** |
| Precision | 98.3%      |
| Recall    | 98.5%      |
| F1 Score  | 98.4%      |

---

## 🚀 Predictive System

A simple prediction script was built using the best model (LinearSVC).

```python
X_new = X_test[0]
prediction = model.predict(X_new)

if prediction[0] == 0:
    print("The news is Real")
else:
    print("The news is Fake")
```

---

## 🚜 How to Run This Project

1. Clone the repository:

```bash
git clone https://github.com/mdadilmuzaffar24/Fake_News_Prediction.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download NLTK stopwords:

```python
import nltk
nltk.download('stopwords')
```

4. Run the notebook or Python script.

---

## 📊 Future Enhancements

* Integrate **deep learning models** (LSTM/GRU) for sequence modeling
* Hyperparameter tuning using **GridSearchCV**
* Build and deploy a web app using **Flask** or **Streamlit**
* Extend the pipeline to support live news feed validation
* Build a REST API endpoint for real-time inference

---

## 👨‍💻 Author

**Md Adil Muzaffar**
M.Tech (CSE - Data Science), Jamia Hamdard University
🔗 [LinkedIn](https://www.linkedin.com/in/md-adil-muzaffar)
💻 [GitHub](https://github.com/mdadilmuzaffar24)

---



## 💖 Badges

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Accuracy](https://img.shields.io/badge/Accuracy-98%25%2B-brightgreen)
![Model](https://img.shields.io/badge/Model-Linear%20SVM-orange)
![NLP](https://img.shields.io/badge/Tech-NLP-blueviolet)

---

# Thank you for visiting 🚀

