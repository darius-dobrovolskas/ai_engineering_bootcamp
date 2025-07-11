import streamlit as st
from qdrant_client import QdrantClient

from chatbot_ui.core.config import config

from chatbot_ui.retrieval import rag_pipeline

qdrant_client = QdrantClient(
    url=f"http://{config.QDRANT_URL}:6333"
)


with st.sidebar:
    st.title("Settings")

    # Temperature slider
    temperature = st.slider("Temperature (randomness)", min_value=0.0, max_value=2.0, value=0.5, step=0.01)
    # Limit of retrieved items
    top_k = st.number_input("Number of retrieved items (top k)", min_value=1, max_value=10, value=5, step=1)

    # Save provider, model, and new settings to session state
    st.session_state.temperature = temperature
    st.session_state.top_k = top_k

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
        output = rag_pipeline(
            prompt,
            qdrant_client=qdrant_client,
            top_k=st.session_state.top_k,
            temperature=st.session_state.temperature
        )
        st.write(output["answer"].answer)
    st.session_state.messages.append({"role": "assistant", "content": output})