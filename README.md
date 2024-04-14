---
title: Text Summarizer
emoji: 🏢
colorFrom: green
colorTo: indigo
sdk: streamlit
sdk_version: 1.29.0
app_file: app.py
pinned: false
license: mit
---

#### New Summarization and NER

News summarization uses "facebook/bart-base" that is fine-tuned using TensorFlow for summarization using 
<a href = "https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail" target="_blank">CNN news articles</a> dataset.<br><br>

The notebook to fine-tune "facebook/bart-base" for news summarization can be found <a href="https://huggingface.co/spaces/Sravan1214/news-summarizer-ner/blob/main/bart_en_summarization.ipynb">here</a>.<br>"# TextCompress" 
