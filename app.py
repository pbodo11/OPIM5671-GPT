import os
import streamlit as st
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# ---------------------------------------------------------
# 1. Page Configuration & Setup
# ---------------------------------------------------------
st.set_page_config(page_title="OPIM5671 Assistant", page_icon="📊")
st.title("📊 OPIM5671 Data Mining Assistant")
st.write("Ask me anything about our MBA Data Analytics course materials!")

# Securely load the Groq API key from Streamlit's secrets
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ---------------------------------------------------------
# 2. Load the Brain (Cached so it only loads once)
# ---------------------------------------------------------
@st.cache_resource
def load_knowledge_base():
    # Make sure this .pkl file is in the same folder as app.py
    with open('OPIM5671_gpt_knowledge_base.pkl', 'rb') as f:
        all_chunks, index = pickle.load(f)
    
    # Load the embedding model (runs fine on Streamlit's CPU)
    embed_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    return all_chunks, index, embed_model

all_chunks, index, embed_model = load_knowledge_base()

# ---------------------------------------------------------
# 3. The RAG Search & Generation Function
# ---------------------------------------------------------
def ask_OPIM5671_gpt(question, k=3):
    # Search the vector database
    query_vec = embed_model.encode([question])
    distances, indices = index.search(np.array(query_vec).astype('float32'), k=k)

    # Gather the text
    retrieved_text = ""
    for i in indices[0]:
        chunk = all_chunks[i].replace("_", " ").replace("$", "")
        retrieved_text += f"\n---\n{chunk}\n"

    # Build the prompt
    prompt = (
        f"You are a rigorous but supportive Teaching Assistant "
        f"for an MBA-level Data Mining and Time Series Forecasting class. "
        f"Your goal is to answer students' questions based on your knowledge base. \n\n"
        f"NOTES: {retrieved_text}"
    )

    # Send to Groq for lightning-fast generation
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ],
        model="mistral-8x7b-32768",
        temperature=0.1,
    )
    
    return chat_completion.choices[0].message.content

# ---------------------------------------------------------
# 4. Streamlit Chat Interface
# ---------------------------------------------------------
# Store the chat history so it doesn't disappear when the page refreshes
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capture user input
if prompt := st.chat_input("Ask a question about Data Mining or Time Series..."):
    
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Generate and show assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching class materials..."):
            answer = ask_OPIM5671_gpt(prompt)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})