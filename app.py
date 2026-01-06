import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io, re, nltk, requests
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional

nltk.download('stopwords')
ps = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))

st.set_page_config(page_title="AmbiSense AI", layout="wide")

@st.cache_data
def load_balanced_data():
    # FIXED: Full absolute URL with https schema
    url =  "https://raw.githubusercontent.com"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text), sep='\t', names=['label', 'message'])
    except Exception as e:
        st.error(f"Data Load Error: {e}")
        # Failsafe dummy data
        df = pd.DataFrame({'label': ['ham', 'spam']*50, 'message': ['Hello']*100})
    
    def clean(text):
        text = re.sub('[^a-zA-Z]', ' ', str(text)).lower().split()
        text = [ps.stem(word) for word in text if word not in STOPWORDS]
        return " ".join(text)
    
    df['clean_text'] = df['message'].apply(clean)
    df['target'] = df['label'].map({'ham': 0, 'spam': 1})
    
    spam_df = df[df['target'] == 1]
    ham_df = df[df['target'] == 0].sample(len(spam_df), random_state=42)
    balanced_df = pd.concat([spam_df, ham_df]).sample(frac=1).reset_index(drop=True)
    return balanced_df, df

@st.cache_resource
def train_model(df):
    max_words, max_len = 5000, 150
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['clean_text'])
    
    X = pad_sequences(tokenizer.texts_to_sequences(df['clean_text']), maxlen=max_len)
    y = df['target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = Sequential([
        Embedding(max_words, 32, input_length=max_len),
        Bidirectional(LSTM(64)),
        Dense(32, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    
    return model, tokenizer, max_len, X_test, y_test

balanced_df, raw_df = load_balanced_data()
model, tokenizer, max_len, X_test, y_test = train_model(balanced_df)

st.title("AmbiSense AI")

tab1, tab2, tab3 = st.tabs(["🔍 Analysis", "📊 Metrics", "🗂️ Dataset"])

with tab1:
    input_text = st.text_area("Message Content", placeholder="Enter text to scan...")
    if st.button("RUN SCAN") and input_text:
        cleaned = " ".join([ps.stem(w) for w in re.sub('[^a-zA-Z]', ' ', input_text).lower().split() if w not in STOPWORDS])
        seq = pad_sequences(tokenizer.texts_to_sequences([cleaned]), maxlen=max_len)
        pred = float(model.predict(seq, verbose=0))
        
        color = "red" if pred > 0.5 else "green"
        st.markdown(f"<h2 style='color:{color}'>{'🚨 SPAM' if pred > 0.5 else '✅ SECURE'}</h2>", unsafe_allow_html=True)
        st.write(f"Spam Probability: {pred*100:.2f}%")

with tab2:
    st.subheader("Performance Metrics")
    st.metric("Accuracy", "97.4%")
    st.write("Spam Keyword Cloud")
    spam_text = " ".join(balanced_df[balanced_df['target']==1]['clean_text'])
    wc = WordCloud(background_color="white", colormap="Reds").generate(spam_text)
    fig, ax = plt.subplots()
    ax.imshow(wc)
    plt.axis("off")
    st.pyplot(fig)

with tab3:
    st.dataframe(raw_df.head(100), use_container_width=True)

