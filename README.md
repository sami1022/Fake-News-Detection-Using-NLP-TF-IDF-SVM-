# Fake News Detection Using NLP

## 📌 Project Overview
Fake news has become a major challenge in the digital age, influencing public opinion and spreading misinformation rapidly.  
This project aims to build a **machine learning-based fake news detection system** that classifies news articles as **Fake** or **Real** using **Natural Language Processing (NLP)** techniques.

The model processes textual news data, extracts meaningful features using **TF-IDF**, and applies a **Support Vector Machine (SVM)** classifier to accurately distinguish between fake and real news articles.

---

## 🎯 Objectives
- To analyze and preprocess raw news text data
- To apply NLP techniques for feature extraction
- To build a classification model for fake news detection
- To evaluate the performance of the model using standard metrics
- To test the model on unseen/custom news input

---

## 📂 Dataset Description
The dataset is sourced from **Kaggle** and consists of two CSV files:

- **Fake.csv** → Contains fake news articles  
- **True.csv** → Contains real news articles  

### Dataset Features:
- `title` – Title of the news article  
- `text` – Full news content  
- `subject` – Category of the news  
- `date` – Date of publication  

A new column `label` is created:
- `0` → Fake News  
- `1` → Real News  

The datasets are merged and shuffled to ensure unbiased training.

---

## 🛠️ Technologies & Tools Used
- **Programming Language:** Python  
- **Development Environment:** Jupyter Notebook  
- **Libraries:**
  - Pandas
  - NumPy
  - Scikit-learn
  - Matplotlib
  - Seaborn
  - Regex (re)

---

## 🔍 Methodology

### 1️⃣ Data Loading
- Load Fake and Real news datasets
- Assign labels and combine them into a single dataset

### 2️⃣ Data Preprocessing
- Convert text to lowercase
- Remove punctuation, numbers, and special characters
- Remove extra whitespaces
- Prepare clean textual data for NLP processing

### 3️⃣ Feature Extraction (NLP)
- **TF-IDF (Term Frequency–Inverse Document Frequency)** is used to convert text into numerical vectors
- Stopwords are removed
- Unigrams and bigrams are included to improve contextual understanding

### 4️⃣ Train-Test Split
- Dataset is split into:
  - 80% Training data
  - 20% Testing data

### 5️⃣ Model Training
- **Support Vector Machine (SVM)** classifier is used
- SVM is effective for high-dimensional text classification tasks

### 6️⃣ Model Evaluation
The model is evaluated using:
- Accuracy Score
- Precision
- Recall
- F1-Score
- Confusion Matrix (visualized using heatmap)

### 7️⃣ Custom News Testing
- The trained model is tested with user-defined news text to predict whether it is fake or real

---

## 📊 Results & Performance
- The SVM model achieves **high accuracy** on the test dataset
- Strong precision and recall values indicate reliable classification
- Confusion matrix confirms balanced performance on both classes

---

## 📁 Project Structure
