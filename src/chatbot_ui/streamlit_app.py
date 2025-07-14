import streamlit as st
import requests

from chatbot_ui.core.config import config

st.set_page_config(
    page_title="Ecommerce Assistant",
    layout="wide",
)

def api_call(method, url, **kwargs):

    def _show_error_popup(message):
        """Show error message as a popup in the top-right corner."""
        st.session_state["error_popup"] = {
            "visible": True,
            "message": message,
        }

    try:
        response = getattr(requests, method)(url, **kwargs)

        try:
            response_data = response.json()
        except requests.exceptions.JSONDecodeError:
            response_data = {"message": "Invalidresponse format from server"}
        
        if response.ok:
            return True, response_data
        
        return False, response_data
    
    except requests.exceptions.ConnectionError:
        _show_error_popup("Connection error. Please check your network connection.")
        return False, {"message": "Connection error"}
    except requests.exceptions.Timeout:
        _show_error_popup("The request timed out. Please try again.")
        return False, {"message": "Request timeout"}
    except requests.exceptions as e:
        _show_error_popup(f"An unexpected error occured: {str(e)}")
        return False, {"message": str(e)}


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
        status, output = api_call("post", f"{config.API_URL}/rag", json={"query": prompt})
        st.write(output.get("answer"))
    st.session_state.messages.append({"role": "assistant", "content": output})