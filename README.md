# Sentiment Analysis for Indonesian Language

Welcome to **Indonesian Sentiment Analysis**! This project implements sentiment analysis using IndoNLU dictionary and applies Machine Learning (ML) techniques for Natural Language Processing (NLP). It follows the ML Life Cycle, including data acquisition, text cleaning, feature engineering, and model evaluation.

## Project Status

🚧 **Status**: `Completed!`

## Features

This project implements two approaches:

1. **Transformer-based Sentiment Analysis (indoBERT)**
2. **SVM-based Sentiment Analysis using TF-IDF**

### Workflow
- **Akuisisi Data**: Data collection from IndoNLU dataset.
- **Text Cleaning dan Pre-processing**: Tokenization and text normalization.
- **Feature Engineering**: TF-IDF vectorization and embeddings.
- **Pengenalan IndoNLU**: Understanding the IndoNLU dataset.
- **Dataset Analisis Sentimen IndoNLU**: Preparing sentiment analysis dataset.
- **Analisis Sentimen dengan Deep Learning**: Implementing a Transformer (indoBERT) model.
- **Konfigurasi dan Load Pre-trained Model**: Configuring and loading indoBERT.
- **Persiapan Dataset Analisis Sentimen**: Data preparation for model training.
- **Uji Model dengan Contoh Kalimat**: Model testing on sample sentences.
- **Fine Tuning dan Evaluasi Prediksi Sentimen**: Fine-tuning models and evaluation.

## Dataset

The dataset for this project can be obtained from:
```bash
git clone https://github.com/indobenchmark/indonlu
```

## Technologies

This project is built using the following libraries:

### Transformer (indoBERT)
```python
import random
import numpy as np
import pandas as pd
import torch
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from nltk.tokenize import TweetTokenizer
from indonlu.utils.forward_fn import forward_sequence_classification
from indonlu.utils.metrics import document_sentiment_metrics_fn
from indonlu.utils.data_utils import DocumentSentimentDataset, DocumentSentimentDataLoader
```

### SVM with TF-IDF
```python
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
import joblib
```

## Setup and Execution

The program is divided into two files, each implementing a different method:
1. **Sentiment_Analysis_IndoNLU.ipynb** (Transformer-based approach using indoBERT)
2. **Sentiment_Analysis_SVM.ipynb** (SVM with TF-IDF approach)

The program can be executed directly from the `.ipynb` files.

## How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repository/sentiment-analysis-indonlu.git
    ```
2. Navigate to the project directory:
    ```bash
    cd sentiment-analysis-indonlu
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the Jupyter Notebook:
    ```bash
    jupyter notebook
    ```

## Conclusion

This project successfully implements sentiment analysis for the Indonesian language using Transformer (indoBERT) and SVM with TF-IDF. The results demonstrate the effectiveness of deep learning and traditional ML techniques in sentiment classification.

---
