# DistilBert for Text Classification
This repository contains a Python implementation for classifying text as offensive or non-offensive using the DistilBERT model, a lightweight version of the BERT model. The code includes data preprocessing, exploratory data analysis, model training, and evaluation.

The primary goal of this project is to fine-tune a pre-trained DistilBERT model to classify text data into two categories: offensive and non-offensive. The workflow includes the following steps:

**Data Loading and Preprocessing:**
The dataset is loaded and preprocessed, including tasks like text cleaning, tokenization, and lemmatization.
Stopwords are removed, and the text is lowercased and stripped of punctuation to prepare it for model input.

**Exploratory Data Analysis (EDA):**
Visualization techniques are applied to understand the distribution of labels and text lengths.
Word clouds, boxplot, and bar plots are generated to display the most frequent words and trigrams in non-offensive text.

**Model Training:**
The dataset is split into training and testing sets (70%, 30% respectively).
A pre-trained DistilBERT model is fine-tuned using the training data.
Training arguments are optimized for faster training and evaluation.

**Model Evaluation:**
The model’s performance is assessed using accuracy, precision, recall, F1 score, ROC AUC, and a confusion matrix.
ROC curves are plotted to visualize the trade-offs between true positive and false positive rates.

**Installation**
To run the code, ensure you have the following dependencies installed:

pip install transformers datasets seaborn matplotlib wordcloud nltk sklearn

Additionally, download the necessary NLTK datasets:

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

**Results**
The model achieves high performance with an ROC AUC of around 0.94 on the training data, demonstrating its ability to distinguish between offensive and non-offensive text effectively.

This project demonstrates the effectiveness of using a pre-trained transformer model like DistilBERT for text classification tasks. The workflow can be easily adapted to other binary or multi-class text classification problems.
