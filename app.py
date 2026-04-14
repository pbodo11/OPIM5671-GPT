import os
import re
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
st.title("📊 OPIM5671 Data Mining and Time Series Forecasting Assistant")
st.write("Ask me anything about our MBA Data Analytics course materials!")

# Securely load the Groq API key from Streamlit's secrets
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ---------------------------------------------------------
# 2. Load the Brain (Cached so it only loads once)
# ---------------------------------------------------------
@st.cache_resource
def load_knowledge_base():
    print("🧠 Loading knowledge base into memory...")
    
    # 1. Load the raw dictionary from the pickle file
    with open('OPIM5671_gpt_knowledge_base.pkl', 'rb') as f:
        data = pickle.load(f)

    # 2. Extract the chunks and the pre-built index directly
    if isinstance(data, dict) and 'chunks' in data and 'index' in data:
        all_chunks = data['chunks']
        # Deserialize the FAISS index that you already built!
        index = faiss.deserialize_index(data['index'])
    else:
        raise ValueError("The .pkl file is not in the expected dictionary format with 'chunks' and 'index'.")

    # 3. Load the embedding model (only used for embedding the user's short questions now)
    embed_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

    print(f"✅ Success! Loaded {len(all_chunks)} chunks and the pre-built FAISS index.")
    return all_chunks, index, embed_model

all_chunks, index, embed_model = load_knowledge_base()

# ---------------------------------------------------------
# 3. The RAG Search & Generation Function
# ---------------------------------------------------------
def ask_OPIM5671_gpt(question, k=3):
    try:
        # 1. Search the vector database
        query_vec = embed_model.encode([question])
        distances, indices = index.search(np.array(query_vec).astype('float32'), k=k)

        # 2. Gather the text
        retrieved_text = ""
        for i in indices[0]:
            chunk = str(all_chunks[i]).replace("_", " ").replace("$", "")
            retrieved_text += f"\n---\n{chunk}\n"
            
        # SAFETY SHIELD: Cap at 15,000 characters
        retrieved_text = retrieved_text[:15000]

        # 3. Build the prompt 
        prompt = f"""You are a rigorous but supportive Teaching Assistant for an MBA-level Data Mining and Time Series Forecasting class. 
        Your goal is to answer students' questions based on your knowledge base.

        ### MATH FORMATTING SCRIPT:
        When generating mathematical equations, you MUST strictly follow this exact LaTeX template:
        $$
        \\begin{{aligned}}
        [Equation Line 1] \\\\
        [Equation Line 2] \\\\
        [Equation Line 3]
        \\end{{aligned}}
        $$
        Do not use single dollar signs. Do not write math as plain text. You must stack every component of an equation neatly on its own line using the double backslash (\\\\) line breaks.
        Always add the description of the terms to your output.

        ### IMAGE DISPLAY RULES:
        1. You DO have the ability to display images to the user.
        2. If the provided context contains an "ASSOCIATED IMAGES:" section with a Markdown image link (e.g., ![Image Name](knowledge_base_images/...)), you MUST include that exact Markdown link in your final response.
        3. NEVER say 'I cannot display images' or 'I don't have the capability to display images'.
        4. Always place the image link on its own line after you explain the concept.
        5. CRITICAL: DO NOT invent, guess, or hallucinate image links. ONLY output a markdown image link if it explicitly exists in the retrieved NOTES provided below.

NOTES:
{retrieved_text}
"""

        # 4. Send to Groq
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=1000, 
        )
        
        return chat_completion.choices[0].message.content
        
    except Exception as e:
        return f"🚨 **API ERROR:** {str(e)}"
    
# ---------------------------------------------------------
# 4. Streamlit Chat Interface 
# ---------------------------------------------------------
# Store the chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            # Extract and render images using the parenthesis-proof regex
            image_paths = re.findall(r'!\[[^\]]*\]\((.*?\.(?:png|jpg|jpeg|gif))\)', message["content"], flags=re.IGNORECASE)
            clean_text = re.sub(r'!\[[^\]]*\]\((.*?\.(?:png|jpg|jpeg|gif))\)', '', message["content"], flags=re.IGNORECASE)
            
            st.markdown(clean_text)
            
            # Draw the images (but only if they actually exist!)
            for img_path in image_paths:
                if os.path.exists(img_path):
                    st.image(img_path)
                else:
                    # The AI hallucinated a fake link, so we quietly ignore it
                    pass
        else:
            st.markdown(message["content"])

# Capture user input
if user_prompt := st.chat_input("Ask a question about Data Mining or Time Series..."):
    
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)
        
    # Generate and show assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching class materials..."):
            answer = ask_OPIM5671_gpt(user_prompt)
            
            # Extract image links 
            # (Make sure these backslashes \ are in your code, they might have gotten lost in chat!)
            image_paths = re.findall(r'!\[[^\]]*\]\((.*?\.(?:png|jpg|jpeg|gif))\)', answer, flags=re.IGNORECASE)
            clean_text = re.sub(r'!\[[^\]]*\]\((.*?\.(?:png|jpg|jpeg|gif))\)', '', answer, flags=re.IGNORECASE)
            
            # Print the clean text
            st.markdown(clean_text)
            
            # 🚨 BULLETPROOF PATHING 🚨
            # 1. Get the exact directory where app.py lives on your computer
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            
            # Draw the images
            for img_path in image_paths:
                # 2. Combine app.py's location with the 'knowledge_base_images/...' string
                full_absolute_path = os.path.join(BASE_DIR, img_path)
                
                # 3. Check the absolute path
                if os.path.exists(full_absolute_path):
                    st.image(full_absolute_path)
                else:
                    # 4. If it fails, print a yellow warning box showing EXACTLY where it looked
                    st.warning(f"⚠️ Debug: Python could not find the file at this exact location:\n {full_absolute_path}")
            
            # Save the raw answer (with the links intact) to history
            st.session_state.messages.append({"role": "assistant", "content": answer})
