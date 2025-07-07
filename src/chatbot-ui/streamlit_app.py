import streamlit as st
from qdrant_client import QdrantClient
from openai import OpenAI
from groq import Groq
from google import genai
from retrieval import rag_pipeline

from core.config import config 

qdrant_client = QdrantClient(
    url=f"http://{config.QDRANT_URL}:6333"
)


with st.sidebar:
    st.title("Settings")
    
    # Dropdown for model selection
    provider = st.selectbox("Select LLM Provider", ["OpenAI", "Google", "GROQ"])
    if provider == "OpenAI":
        model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"])
    elif provider == "Google":
        model_name = st.selectbox("Model", ["gemini-2.0-flash"])
    elif provider == "GROQ":
        model_name = st.selectbox("Model", ["llama-3.3-70b-versatile"])

    # Temperature slider
    temperature = st.slider("Temperature (randomness)", min_value=0.0, max_value=2.0, value=0.5, step=0.01)
    # Max tokens input
    max_tokens = st.number_input("Max Tokens", min_value=1, max_value=700, value=500, step=1)
    # Limit of retrieved items
    top_k = st.number_input("Number of retrieved items (top k)", min_value=1, max_value=10, value=5, step=1)

    # Save provider, model, and new settings to session state
    st.session_state.provider = provider    
    st.session_state.model_name = model_name
    st.session_state.temperature = temperature
    st.session_state.max_tokens = max_tokens
    st.session_state.top_k = top_k

if st.session_state.get("provider") == "OpenAI":
    client = OpenAI(api_key=config.OPEN_AI_KEY)
elif st.session_state.get("provider") == "GROQ":
    client = Groq(api_key=config.GROQ_AI_KEY)
else:
    client = genai.Client(api_key=config.GOOGLE_AI_KEY)

def run_llm(client, messages, max_tokens=500, temperature=1.0):
    if st.session_state.provider == "Google":
        return client.models.generate_content(
            model=st.session_state.model_name,
            contents=[message["content"] for message in messages],
            config=genai.types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                )
        ).text
    else:
        return client.chat.completions.create(
            model=st.session_state.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        ).choices[0].message.content

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "Hello! How can I assist you today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hello! How can I assist you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # output = run_llm(
        #     client,
        #     st.session_state.messages,
        #     max_tokens=st.session_state.max_tokens,
        #     temperature=st.session_state.temperature
        # )
        output = rag_pipeline(
            prompt,
            qdrant_client=qdrant_client,
            top_k=st.session_state.top_k,
            temperature=st.session_state.temperature
        )
        st.write(output["answer"])
    st.session_state.messages.append({"role": "assistant", "content": output})