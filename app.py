import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re, io, requests, nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional

st.set_page_config(page_title="AmbiSense AI", layout="wide")

nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))
ps = PorterStemmer()

@st.cache_data
def load_balanced_data():
    DATA_URL = (
        "https://raw.githubusercontent.com/"
        "justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    )

    try:
        response = requests.get(DATA_URL, timeout=10)
        response.raise_for_status()
        df = pd.read_csv(
            io.StringIO(response.text),
            sep="\t",
            names=["label", "message"]
        )
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return pd.DataFrame(), pd.DataFrame()

    def clean_text(text):
        text = re.sub("[^a-zA-Z]", " ", str(text)).lower().split()
        text = [ps.stem(w) for w in text if w not in STOPWORDS]
        return " ".join(text)

    df["clean_text"] = df["message"].apply(clean_text)
    df["target"] = df["label"].map({"ham": 0, "spam": 1})

    spam_df = df[df["target"] == 1]
    ham_df = df[df["target"] == 0]

    if spam_df.empty or ham_df.empty:
        st.error("Dataset contains only one class. Cannot train model.")
        return pd.DataFrame(), df

    ham_df = ham_df.sample(len(spam_df), random_state=42)
    balanced_df = (
        pd.concat([spam_df, ham_df])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    return balanced_df, df

@st.cache_resource
def train_model(df):
    if df.empty:
        return None, None, None

    max_words = 5000
    max_len = 150

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["clean_text"])

    X = pad_sequences(
        tokenizer.texts_to_sequences(df["clean_text"]),
        maxlen=max_len
    )
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Sequential([
        Embedding(max_words, 32, input_length=max_len),
        Bidirectional(LSTM(64)),
        Dense(32, activation="relu"),
        Dropout(0.4),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_train,
        y_train,
        epochs=5,
        batch_size=32,
        verbose=0
    )

    return model, tokenizer, max_len

balanced_df, raw_df = load_balanced_data()

if balanced_df.empty:
    st.stop()

model, tokenizer, max_len = train_model(balanced_df)

if model is None:
    st.stop()

st.title("AmbiSense AI")

tab1, tab2, tab3 = st.tabs(["🔍 Analysis", "📊 Metrics", "🗂️ Dataset"])

with tab1:
    input_text = st.text_area("Message Content", placeholder="Enter text to scan...")

    if st.button("RUN SCAN") and input_text.strip():
        cleaned = " ".join([
            ps.stem(w)
            for w in re.sub("[^a-zA-Z]", " ", input_text).lower().split()
            if w not in STOPWORDS
        ])

        seq = pad_sequences(
            tokenizer.texts_to_sequences([cleaned]),
            maxlen=max_len
        )

        pred = float(model.predict(seq, verbose=0)[0][0])

        if pred > 0.5:
            st.markdown("<h2 style='color:red'>🚨 SPAM</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='color:green'>✅ SECURE</h2>", unsafe_allow_html=True)

        st.write(f"Spam Probability: **{pred * 100:.2f}%**")

with tab2:
    st.subheader("Model Insights")
    st.metric("Model Type", "BiLSTM")
    st.metric("Dataset Size", len(balanced_df))

    spam_text = " ".join(
        balanced_df[balanced_df["target"] == 1]["clean_text"]
    )

    wc = WordCloud(
        background_color="white",
        colormap="Reds",
        width=800,
        height=400
    ).generate(spam_text)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wc)
    ax.axis("off")
    st.pyplot(fig)

with tab3:
    st.dataframe(raw_df.head(100), use_container_width=True)
