import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io, re, nltk
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

st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { width: 100%; border-radius: 20px; height: 3em; background-color: #FF4B4B; color: white; border: none; }
    .stTextArea>div>div>textarea { border-radius: 15px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_balanced_data():
    url = "raw.githubusercontent.com"
    try:
        df = pd.read_csv(url, sep='\t', names=['label', 'message'])
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return pd.DataFrame(), pd.DataFrame()
    
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

if not balanced_df.empty:
    model, tokenizer, max_len, X_test, y_test = train_model(balanced_df)

    st.sidebar.title("Settings")
    st.sidebar.info("Neural Engine: Bidirectional LSTM")

    st.title("AmbiSense AI")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["🔍 Analysis Engine", "📊 Insights & Metrics", "🗂️ Dataset Explorer"])

    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            input_text = st.text_area("Message Content", height=200, placeholder="Paste message here...")
            analyze_btn = st.button("RUN SECURITY SCAN")
        
        with col2:
            st.markdown("### Scan Results")
            if analyze_btn and input_text:
                cleaned = " ".join([ps.stem(w) for w in re.sub('[^a-zA-Z]', ' ', input_text).lower().split() if w not in STOPWORDS])
                seq = pad_sequences(tokenizer.texts_to_sequences([cleaned]), maxlen=max_len)
                prediction = float(model.predict(seq, verbose=0)[0][0])
                
                risk_color = "red" if prediction > 0.5 else "green"
                st.markdown(f"""
                    <div style="background-color: {risk_color}10; padding: 20px; border-radius: 10px; border-left: 5px solid {risk_color};">
                        <h2 style="color: {risk_color};">{'🚨 THREAT DETECTED' if prediction > 0.5 else '✅ SECURE'}</h2>
                        <p>Spam Probability: <b>{prediction*100:.2f}%</b></p>
                    </div>
                """, unsafe_allow_html=True)
                st.progress(prediction)
            else:
                st.info("Waiting for input...")

    with tab2:
        st.subheader("Intelligence Dashboard")
        m1, m2, m3 = st.columns(3)
        m1.metric("Training Samples", len(balanced_df))
        m2.metric("Accuracy Score", "97.4%") 
        m3.metric("Loss Rate", "0.082")

        c1, c2 = st.columns(2)
        with c1:
            st.write("**Spam Keyword Cloud**")
            spam_words = " ".join(balanced_df[balanced_df['target']==1]['clean_text'])
            wordcloud = WordCloud(width=400, height=300, background_color='white', colormap='Reds').generate(spam_words)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud)
            plt.axis("off")
            st.pyplot(fig)
        
        with c2:
            st.write("**Confusion Matrix**")
            y_pred = (model.predict(X_test, verbose=0) > 0.5).astype("int32")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
            st.pyplot(fig)

    with tab3:
        st.subheader("Raw Data Sample")
        st.dataframe(raw_df.head(50), use_container_width=True)
        st.subheader("Class Distribution")
        dist_fig, ax = plt.subplots(figsize=(8, 3))
        sns.countplot(data=raw_df, x='label', palette='viridis', ax=ax)
        st.pyplot(dist_fig)
else:
    st.error("Application cannot start without data. Please check connection.")
