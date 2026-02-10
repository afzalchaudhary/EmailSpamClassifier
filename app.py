import streamlit as st
import nltk
import pickle
import os
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

@st.cache_resource
def setup_nltk():
    nltk.download('punkt')
    nltk.download('stopwords')

setup_nltk()

ps = PorterStemmer()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)

  y=[]
  for i in text:
    if i.isalnum():
      y.append(i)
  text=y[:]
  y.clear()
  for i in text:
    if i not in stopwords.words("english") and i not in string.punctuation:
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    y.append(ps.stem(i))

  return " " .join(y)


vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")
model_path = os.path.join(BASE_DIR, "model.pkl")

tfidf = pickle.load(open(vectorizer_path, "rb"))
model = pickle.load(open(model_path, "rb"))
# tfidf = pickle.load(open('vectorizer.pkl','rb'))
# model = pickle.load(open('model.pkl','rb'))

st.title("Email Spam Classifier")

# input_email=st.text_input("Enter the Message")
input_email = st.text_area("Enter the Message", height=200)

if st.button("Predict"):
    # 1. preprocess
    transform_email=transform_text(input_email)
    # 2. vectorize
    vector_input=tfidf.transform([transform_email])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display

    if result == 1:
        st.header("🚨 Spam")
    else:
        st.header("✅ Not Spam")


