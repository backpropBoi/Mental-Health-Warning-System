# Mental-Health-Warning-System
Mental Health Text Classification: Depression Detection from Reddit Posts
<br>

This repository contains a machine learning project focused on detecting depression from Reddit posts using natural language processing (NLP) and deep learning techniques. The goal is to build and evaluate models that can classify whether a given Reddit post indicates depression, leveraging a real-world, anonymized dataset.

<br>

Project Overview
<br>
This project aims to identify signs of depression in Reddit posts using machine learning. By analyzing user-generated text, the models attempt to classify whether a post reflects depressive symptoms. The main motivation is to explore how NLP and deep learning can support early detection of mental health issues in online communities.
<br>

Dataset
<br>
1.Source: The dataset was taken from hugging face( hugginglearners/reddit-depression-cleaned)<br>

2.Description: Each row contains a Reddit post and a binary label (is_depression), where 1 indicates the post is related to depression and 0 otherwise.<br>

3.Preprocessing: Text is cleaned and prepared for analysis (e.g., removing special characters, lowercasing).
<br>

Project Structure:
<br>
The dataset, containing cleaned Reddit posts and depression labels, is loaded and split into training and testing sets. Text data is then preprocessed and tokenized using a BERT tokenizer. A custom PyTorch Dataset class is defined to handle tokenized text and labels, and DataLoaders are created for efficient batching. The core model is a BERT-LSTM neural network, which uses frozen BERT embeddings followed by a bidirectional LSTM and a dense output layer with sigmoid activation for binary classification. The model is trained using binary cross-entropy loss and the Adam optimizer over several epochs, with loss tracked per epoch. After training, the model is evaluated on the test set, reporting metrics such as accuracy, F1 score, and a detailed classification report. The code also generates and visualizes a confusion matrix and ROC curve to assess performance, and includes functionality for plotting precision-recall curves.The code also has a comparitive analysis of the BERT-LSTM model with naive bayes, KNN and decision tree algorithms.

