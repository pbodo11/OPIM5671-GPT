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
    # 1. Load the raw data from the pickle file
    with open('OPIM5671_gpt_knowledge_base.pkl', 'rb') as f:
        data = pickle.load(f)

    # 2. Smartly extract the chunks no matter how they were saved!
    if isinstance(data, list):
        # If it was saved purely as a list
        all_chunks = data
    elif isinstance(data, tuple):
        # If it was saved as a tuple
        all_chunks = data[0]
    elif isinstance(data, dict):
        # If it was saved as a dictionary, find the list inside it
        for key, value in data.items():
            if isinstance(value, list):
                all_chunks = value
                break

    # 3. Load the embedding model
    embed_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

    # 4. REBUILD THE FAISS INDEX
    embeddings = embed_model.encode(all_chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))

    return all_chunks, index, embed_model

all_chunks, index, embed_model = load_knowledge_base()

# ---------------------------------------------------------
# 3. The RAG Search & Generation Function (DIAGNOSTIC MODE)
# ---------------------------------------------------------
def ask_OPIM5671_gpt(question, k=3):
    try:
        # 1. Search the vector database (Just to make sure this part doesn't crash)
        query_vec = embed_model.encode([question])
        distances, indices = index.search(np.array(query_vec).astype('float32'), k=k)

        # 2. Bypass the heavy text for a moment
        prompt = "You are a helpful test assistant. Ignore the user's question and just reply: 'The API is working flawlessly!'"

        # 3. Send a tiny, incredibly safe request to Groq
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question}
            ],
            model="llama3-8b-8192", # The smallest, fastest model
            temperature=0.1,
            max_tokens=100, 
        )
        
        return chat_completion.choices[0].message.content
        
    except Exception as e:
        # If Groq crashes, this will print the REAL error directly into your chat window!
        return f"🚨 **GROQ API ERROR:** {str(e)}"
    
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
