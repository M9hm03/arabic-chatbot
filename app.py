import streamlit as st
import pandas as pd
import json
import pickle
import os
import spacy
import nltk
from word2number import w2n
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator
from transformers import pipeline
import openai

# ---------- SETUP ----------
st.set_page_config(page_title="Arabic Chatbot", layout="centered")
st.title("🧠 Arabic AI Chatbot")

# Initialize session state if not already
if "phase" not in st.session_state:
    st.session_state.phase = "collect"
    st.session_state.user_info = {
        "name": None,
        "gender": None,
        "marital_status": None,
        "education_level": None,
        "income_group": None,
        "age": None,
        "smoker_now": None,
        "years_of_smoking": None,
        "cigarettes_per_day": None,
        "passive_smoker": None,
        "family_history_recorded_cases": None,
        "occupational_exposure": None,
        "healthcare_access": None,
        "country": None
    }
    st.session_state.ds1_user_info = st.session_state.user_info.copy()
    st.session_state.messages = []

# ---------- LOAD DATA ----------
data_path = 'data'  # Use relative path; put all your files in ./data/

@st.cache_resource(show_spinner=False)
def load_resources():
    females_ar = pd.read_csv(os.path.join(data_path, 'females_ar.csv'))
    males_ar = pd.read_csv(os.path.join(data_path, 'males_ar.csv'))
    with open(os.path.join(data_path, 'education.json'), encoding='utf-8') as f:
        education_patterns = json.load(f)
    with open(os.path.join(data_path, 'martial_status_male.json'), encoding='utf-8') as f:
        marital_male = json.load(f)
    with open(os.path.join(data_path, 'martial_status_female.json'), encoding='utf-8') as f:
        marital_female = json.load(f)
    with open(os.path.join(data_path, 'ds3_model.pkl'), 'rb') as f:
        ds3_model = pickle.load(f)
    with open(os.path.join(data_path, 'lung_cancer_model.pkl'), 'rb') as f:
        lung_model = pickle.load(f)
    return females_ar, males_ar, education_patterns, marital_male, marital_female, ds3_model, lung_model

females_ar, males_ar, education_patterns, marital_male, marital_female, ds3_model, lung_cancer_model = load_resources()

# ---------- NLP + TRANSLATION ----------
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nlp = spacy.load("en_core_web_sm")
zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def translate_to_english(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception as e:
        return f"Translation error: {e}"

def translate_to_arabic(text):
    try:
        return GoogleTranslator(source='auto', target='ar').translate(text)
    except Exception as e:
        return f"Translation error: {e}"

def cardinal_ner(text):
    translated = translate_to_english(text)
    doc = nlp(translated)
    for ent in doc.ents:
        if ent.label_ == "CARDINAL":
            try:
                return int(w2n.word_to_num(ent.text))
            except:
                continue
    try:
        return int(w2n.word_to_num(translated))
    except:
        return text

def map_education_level(response):
    translated = translate_to_english(response).lower()
    best_tag, best_score = None, 0
    for tag, patterns in education_patterns.items():
        for pattern in patterns:
            score = cosine_similarity(
                TfidfVectorizer().fit_transform([translated, pattern.lower()])
            )[0, 1]
            if score > best_score:
                best_score = score
                best_tag = tag
    return best_tag

def map_marital_status(response, gender):
    translated = translate_to_english(response).lower()
    dataset = marital_male if gender == 'male' else marital_female
    best_tag, best_score = None, 0
    for tag, patterns in dataset.items():
        for pattern in patterns:
            score = cosine_similarity(
                TfidfVectorizer().fit_transform([translated, pattern.lower()])
            )[0, 1]
            if score > best_score:
                best_score = score
                best_tag = tag
    return best_tag

def classify_yes_no(text):
    translated = translate_to_english(text)
    result = zero_shot_classifier(translated, candidate_labels=["yes", "no"])
    return result['labels'][0]

# ---------- UI PHASE 1: COLLECT BASIC INFO ----------
if st.session_state.phase == "collect":
    st.subheader("الرجاء إدخال المعلومات التالية")
    name = st.text_input("ما اسمك؟")
    if name:
        st.session_state.user_info["name"] = name
        st.session_state.user_info["gender"] = st.selectbox("ما جنسك؟", ["male", "female"])
        marital = st.text_input("ما حالتك الاجتماعية؟")
        st.session_state.user_info["marital_status"] = map_marital_status(marital, st.session_state.user_info['gender'])

        edu = st.text_input("ما مستواك التعليمي؟")
        st.session_state.user_info["education_level"] = map_education_level(edu)

        income = st.text_input("ما هو دخلك؟")
        st.session_state.user_info["income_group"] = cardinal_ner(income)

        age = st.text_input("ما هو عمرك؟")
        st.session_state.user_info["age"] = cardinal_ner(age)

        st.session_state.ds1_user_info = st.session_state.user_info.copy()

        if all(st.session_state.user_info.values()):
            st.success("تم جمع البيانات بنجاح! يمكنك الآن بدء المحادثة.")
            if st.button("بدء المحادثة"):
                st.session_state.phase = "chat"
                st.rerun()

# ---------- PHASE 2: GPT CHAT MODE ----------
elif st.session_state.phase == "chat":
    st.subheader("💬 تحدث مع الذكاء الاصطناعي")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("أدخل سؤالك بالعربية..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        # GPT API call
        openai.api_key = st.secrets["openai_api_key"]
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "أنت مساعد طبي يتحدث العربية."},
                    *st.session_state.messages
                ]
            )
            reply = completion.choices[0].message.content
        except Exception as e:
            reply = f"حدث خطأ: {str(e)}"

        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

    # 3-dot menu
    with st.expander("⋯ خيارات إضافية"):
        if st.button("اظهر نسبه ميلاني للتدخين"):
            st.write("نسبة الميل للتدخين:")
            st.write(ds3_model.predict([list(st.session_state.user_info.values())])[0])

        if st.button("خطر الاصابة بسرطان الرئة"):
            st.write("نسبة الخطر:")
            st.write(lung_cancer_model.predict([list(st.session_state.user_info.values())])[0])

        if st.button("داتا 3"):
            st.write("محتوى الداتا 3:")
            st.write(st.session_state.ds1_user_info)
