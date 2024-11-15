import streamlit as st
from utils import preprocess_text, classify_text_with_lda, transform_with_bert, classify_text_with_kmeans


st.title('Clasificación de Texto')

# Entrada de texto por el usuario
input_text = st.text_area("Ingrese una frase para clasificar:")

# Selección del modelo
model_option = st.selectbox("Seleccione el modelo para clasificar:", ("LDA con TFIDF", "LDA sin TFIDF", "KMeans con BERT"))

# Botón para clasificar
if st.button("Clasificar"):
    if not input_text:
        st.error("Por favor, ingrese una frase para clasificar.")
    else:
        # Preprocesamiento de la frase
        processed_text = preprocess_text(input_text)

        if model_option == "LDA con TFIDF":
            # Clasificación usando LDA con TFIDF
            topic_name, score = classify_text_with_lda(processed_text, use_tfidf=True)
            st.write(f"La frase fue clasificada en el tema: {topic_name}")


        elif model_option == "LDA sin TFIDF":
            # Clasificación usando LDA sin TFIDF
            topic_name, score = classify_text_with_lda(processed_text, use_tfidf=False)
            st.write(f"La frase fue clasificada en el tema: {topic_name}")


        elif model_option == "KMeans con BERT":
            # Transformar el texto usando BERT
            text_embedding = transform_with_bert(input_text)
            # Clasificación usando KMeans
            cluster_name = classify_text_with_kmeans(text_embedding)
            st.write(f"La frase fue clasificada en el clúster: {cluster_name}")

