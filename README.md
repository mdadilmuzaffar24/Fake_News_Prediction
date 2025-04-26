# 📰 Fake News Detection Using Machine Learning

---

## 📛 Problem Statement

In the current digital era, the spread of fake news through social media and online platforms has become a major challenge. Fake news can influence public opinion, create panic, or manipulate markets and societies. There is a strong need for an automated system that can efficiently classify news articles as **real** or **fake** to ensure information integrity.

---

## 🌟 Objective

- To develop a Machine Learning model capable of classifying news articles as **real** or **fake**.
- To apply **Natural Language Processing (NLP)** techniques to pre-process textual data.
- To evaluate the model’s performance using accuracy metrics on both training and testing datasets.
- To build a simple **predictive system** that can predict the authenticity of new news articles.

---

## 📚 About the Dataset

| Column | Description |
|:---|:---|
| **id** | Unique identifier for each news article |
| **title** | Title of the news article |
| **author** | Author of the news article |
| **text** | Main text body of the news article (may be incomplete) |
| **label** | Target label - **0** (Real News) or **1** (Fake News) |

- Dataset Shape: **(20800, 5)**

---

## 🔧 Technologies and Libraries Used

- **Python 3**
- **Pandas** – Data analysis and manipulation
- **NumPy** – Numerical operations
- **Regular Expressions (re)** – Text cleaning
- **NLTK** (Natural Language Toolkit) – Stopwords removal and Stemming
- **Scikit-learn** – ML modeling and evaluation:
  - TfidfVectorizer (Text to numerical feature transformation)
  - LogisticRegression (Model building)
  - Train-Test Split
  - Accuracy Score (Model evaluation)

---

## 🛠️ Project Workflow

### 1. Importing the Dependencies

- Imported essential libraries for data handling, text processing, feature extraction, and model building.

---

### 2. Data Preprocessing

- **Loading the Dataset**
  - Loaded the dataset into a Pandas DataFrame and examined the shape and contents.

- **Handling Missing Values**
  - Replaced missing values in `author`, `title`, and `text` with empty strings.

- **Merging Important Features**
  - Combined `author` and `title` into a single field called `content`.

- **Text Cleaning & Stemming**
  - Converted text to lowercase
  - Removed special characters
  - Removed stopwords
  - Applied **Porter Stemming**

- **Feature Transformation**
  - Used **TfidfVectorizer** to convert cleaned text into numerical feature vectors.

---

### 3. Model Building

- **Train-Test Split**
  - Divided the dataset into training (80%) and testing (20%) sets.
  - Used stratified splitting to maintain class balance.

- **Training the Model**
  - Trained a **Logistic Regression** model on the TF-IDF feature vectors.

---

### 4. Model Evaluation

| Metric | Score |
|:---|:---|
| **Training Accuracy** | 98.63% |
| **Testing Accuracy** | 97.91% |

- Achieved high accuracy on both datasets, demonstrating excellent generalization.

---

### 5. Predictive System

- Built a predictive system to classify new/unseen news articles as Real or Fake.

Example Output:
```bash
Prediction: 1
The news is Fake
```
Verification:
```bash
True label: 1
```

---

## 🚀 How to Run This Project

1. Clone this repository.
2. Install required libraries:
```bash
pip install pandas numpy scikit-learn nltk
```
3. Download NLTK stopwords:
```python
import nltk
nltk.download('stopwords')
```
4. Run the notebook or script.

---

## 📊 Future Work

- Experiment with advanced models like **Random Forest**, **XGBoost**, **SVM**, or **Deep Learning** (LSTM/GRU).
- Hyperparameter tuning for improved performance.
- Deploy a **web application** using **Streamlit** or **Flask**.
- Build a REST API for fake news detection.

---

## 👨‍💻 Author

**Md Adil Muzaffar**

- 🔗 [LinkedIn Profile](https://www.linkedin.com/in/md-adil-muzaffar)
- 💻 GitHub: [yourusername]([https://github.com/yourusername](https://github.com/mdadilmuzaffar24))

---

## 💚 Badges

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Accuracy](https://img.shields.io/badge/Accuracy-98%25%2B-brightgreen)
![Model](https://img.shields.io/badge/Model-Logistic%20Regression-orange)
![NLP](https://img.shields.io/badge/Tech-NLP-blueviolet)

---

# Thank You! 🚀

