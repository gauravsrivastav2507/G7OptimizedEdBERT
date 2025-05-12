
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from wordcloud import WordCloud
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch

# Title
st.title("SortED: Student Upgrade Predictor & Concern Analyzer")

# Upload
uploaded_file = st.file_uploader("Upload your CSV file with 'student_review' and 'UP' columns", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    if 'student_review' not in data.columns or 'UP' not in data.columns:
        st.error("Missing required columns: 'student_review' and 'UP'")
    else:
        st.success("File loaded successfully!")

        data['student_review'] = data['student_review'].fillna('No review provided.')
        data['target'] = data['UP']

        st.write("### Sample Data")
        st.dataframe(data[['student_review', 'UP']].head())

        # Tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        train_texts = data['student_review'].astype(str).tolist()
        train_labels = data['target'].tolist()
        encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)

        class ReviewDataset(Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels
            def __getitem__(self, idx):
                return {
                    'input_ids': self.encodings['input_ids'][idx],
                    'attention_mask': self.encodings['attention_mask'][idx],
                    'labels': torch.tensor(self.labels[idx], dtype=torch.long)
                }
            def __len__(self):
                return len(self.labels)

        dataset = ReviewDataset(encodings, train_labels)

        # BERT Model
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=1,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            logging_dir='./logs',
            logging_steps=10,
            save_strategy="no"
        )

        trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
        trainer.train()

        # Predictions
        preds = trainer.predict(dataset)
        y_pred = np.argmax(preds.predictions, axis=1)
        y_true = preds.label_ids

        st.subheader("📊 Model Evaluation")
        st.write("**Accuracy:**", accuracy_score(y_true, y_pred))
        st.text(classification_report(y_true, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # Extract concerns from failed reviews
        st.subheader("⚠️ Concern Keyword Extraction")
        failed_reviews = data.loc[data['target'] == 0, 'student_review'].astype(str).str.lower()

        def tokenize(text):
            text = re.sub(r'[^\w\s]', '', text)
            words = text.split()
            return [w for w in words if w not in ENGLISH_STOP_WORDS and len(w) > 2]

        all_words = []
        for review in failed_reviews:
            all_words.extend(tokenize(review))

        concern_freq = Counter(all_words).most_common(20)
        concern_df = pd.DataFrame(concern_freq, columns=["Concern", "Frequency"])
        st.dataframe(concern_df)

        # Wordcloud
        st.subheader("☁️ WordCloud of Top Concerns")
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(all_words))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

        # Concern impact
        st.subheader("📉 Impact of Keywords on Upgrade")
        keywords = [kw for kw, _ in concern_freq[:10]]
        for word in keywords:
            data[f'{word}_mention'] = data['student_review'].str.lower().str.contains(rf'\b{word}\b').astype(int)
        correlations = data[[f'{word}_mention' for word in keywords] + ['target']].corr()['target'][:-1].sort_values()
        st.bar_chart(correlations)
