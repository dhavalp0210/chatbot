import streamlit as st
import io
import random
import string
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer

# Ensure all necessary NLTK data is downloaded
nltk.download('punkt', quiet=True)  # Download 'punkt' for tokenization
nltk.download('wordnet', quiet=True)  # Download 'wordnet' for lemmatization
nltk.download('stopwords', quiet=True)  # Download stopwords if required later

# Reading in the corpus
with open('chatbot.txt', 'r', encoding='utf8', errors='ignore') as fin:
    raw = fin.read().lower()

# Tokenization
sent_tokens = nltk.sent_tokenize(raw)  # Converts to list of sentences
word_tokens = nltk.word_tokenize(raw)  # Converts to list of words

# Preprocessing
lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Keyword Matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Generating response
def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robo_response = robo_response + "I am sorry! I don't understand you."
    else:
        robo_response = robo_response + sent_tokens[idx]
    sent_tokens.remove(user_response)
    return robo_response

# Streamlit App
st.title("ICICI Chatbot")
st.write("I am a helpful assistant. Ask me about ICICI bank!")

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Input from user
user_input = st.text_input("Type your message", "", key="input")

if user_input:
    user_input = user_input.lower()
    if user_input != 'bye':
        if user_input in ('thanks', 'thank you'):
            st.session_state['messages'].append(("You", user_input))
            st.session_state['messages'].append(("Chatbot", "You are welcome."))
        else:
            if greeting(user_input) is not None:
                bot_response = greeting(user_input)
            else:
                bot_response = response(user_input)
            st.session_state['messages'].append(("You", user_input))
            st.session_state['messages'].append(("Chatbot", bot_response))
    else:
        st.session_state['messages'].append(("You", user_input))
        st.session_state['messages'].append(("Chatbot", "Bye! Take care.."))

# Display chat history in WhatsApp-like style
st.markdown("<style>" \
            "div[data-testid=\"stVerticalBlock\"] {" \
            "    border: 1px solid #d1d1d1;" \
            "    border-radius: 10px;" \
            "    padding: 10px;" \
            "    margin: 5px 0;" \
            "}" \
            "div[data-testid=\"stVerticalBlock\"]:nth-child(even) {" \
            "    background-color: #dcf8c6;" \
            "}" \
            "div[data-testid=\"stVerticalBlock\"]:nth-child(odd) {" \
            "    background-color: #ffffff;" \
            "}" \
            "</style>", unsafe_allow_html=True)

for sender, message in st.session_state['messages']:
    if sender == "You":
        st.markdown(f"<div data-testid=\"stVerticalBlock\"><b>You:</b> {message}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div data-testid=\"stVerticalBlock\"><b>Chatbot:</b> {message}</div>", unsafe_allow_html=True)
