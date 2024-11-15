# utils.py

import pickle
import gensim
from gensim import corpora, models
from gensim.parsing.preprocessing import STOPWORDS
from transformers import BertTokenizer, BertModel
import torch
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# Cargar los modelos y el diccionario necesarios
lda_model = gensim.models.LdaModel.load("lda_model.model")  # LDA sin TF-IDF
lda_model_tfidf = gensim.models.LdaModel.load("lda_modeltfidf.model")  # LDA con TF-IDF
kmeans_model = pickle.load(open('kmeans_model.pkl', 'rb'))
dictionary = corpora.Dictionary.load("dictionary.dict")  # Diccionario

# Mapeo de clústeres a temas
cluster_mapping = { 
    0: 'Crime',
    1: 'Government',
    2: 'Council Report'
}

# Cargar modelo de lenguaje para lematización
nlp = spacy.load("en_core_web_lg")

# Inicializar el vectorizador TFIDF
tfidf_vectorizer = TfidfVectorizer()

# Función para preprocesar la frase
def preprocess_text(text):
    doc = nlp(text.lower())
    processed_text = [
        token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in STOPWORDS
    ]
    return processed_text

# Función para convertir la frase a su representación de BoW o TF-IDF
def get_representation(processed_text, use_tfidf):
    bow_vector = dictionary.doc2bow(processed_text)
    if use_tfidf:
        tfidf = models.TfidfModel(dictionary=dictionary)
        return tfidf[bow_vector]
    return bow_vector

# Clasificación de la frase usando el modelo LDA
def classify_text_with_lda(processed_text, use_tfidf):
    representation = get_representation(processed_text, use_tfidf)
    model = lda_model_tfidf if use_tfidf else lda_model
    topics = model[representation]
    topics_sorted = sorted(topics, key=lambda x: x[1], reverse=True)
    topic, score = topics_sorted[0]
    return cluster_mapping.get(topic, "Unknown"), score

# Inicializar el tokenizer y el modelo BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Función para transformar el texto en embeddings usando BERT
def transform_with_bert(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings

# Función para clasificar usando KMeans con BERT
def classify_text_with_kmeans(text_embedding):
    predicted_class = kmeans_model.predict(text_embedding)[0]
    return cluster_mapping.get(predicted_class, "Unknown")
