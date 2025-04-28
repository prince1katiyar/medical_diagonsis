


import streamlit as st
import requests
import io
from PIL import Image
import base64
from dotenv import load_dotenv
import os
import chromadb
from sentence_transformers import SentenceTransformer
import uuid
import pandas as pd
from datetime import datetime

load_dotenv()

st.set_page_config(page_title="Medical Imaging Diagnosis", page_icon="🏥", layout="wide")
st.title("🏥 Medical Imaging Diagnosis Agent")
st.caption("AI-driven medical image analysis integrated with OpenAI API and semantic search")

API_URL = "http://localhost:8000/analyze"

# ChromaDB setup
client = chromadb.PersistentClient(path="./medical_db")
collection = client.get_or_create_collection("medical_analyses")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("OpenAI API Key", type="password")
    if api_key:
        st.session_state.OPENAI_API_KEY = api_key
        st.success("API Key configured!")

uploaded_file = st.file_uploader("Upload Medical Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    aspect_ratio = image.height / image.width
    resized_image = image.resize((500, int(500 * aspect_ratio)))
    st.image(resized_image, caption="Uploaded Medical Image")

    if st.button("🔍 Analyze"):
        if not api_key:
            st.error("Please enter your API key in the sidebar.")
        else:
            with st.spinner("Analyzing image..."):
                buffered = io.BytesIO()
                resized_image.save(buffered, format='PNG')
                encoded_img = base64.b64encode(buffered.getvalue()).decode()

                response = requests.post(
                    API_URL,
                    json={"image": encoded_img, "filename": uploaded_file.name, "api_key": api_key},
                    timeout=120
                )

                if response.ok:
                    data = response.json()
                    st.subheader("📋 Analysis Results")
                    st.markdown(data["analysis"])
                    if data.get("references"):
                        with st.expander("📚 References"):
                            st.markdown(data["references"])

                    # Store analysis in ChromaDB
                    embedding = embedding_model.encode(data["analysis"]).tolist()
                    collection.add(
                        ids=[str(uuid.uuid4())],
                        embeddings=[embedding],
                        documents=[data["analysis"]],
                        metadatas=[{"filename": uploaded_file.name, "date": datetime.now().isoformat()}]
                    )
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
else:
    st.info("👆 Upload a medical image to analyze.")

# Semantic search interface
st.markdown("---")
st.header("🔎 Semantic Search in Past Analyses")
query = st.text_input("Ask a question about past analyses")
if query and st.button("Search"):
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=5)

    if results["documents"]:
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            st.subheader(f"Result from {meta['filename']} ({meta['date']})")
            st.markdown(doc)
    else:
        st.info("No relevant past analyses found.")
        