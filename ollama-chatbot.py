import streamlit as st
import requests
import json

# Set page configuration
st.set_page_config(page_title="Ollama Chatbot", page_icon="🤖", layout="wide")

# Ollama API endpoint
OLLAMA_API_URL = "https://5c3e-34-82-129-29.ngrok-free.app"

def query_ollama(prompt, model="mistral", system_prompt=None):
    """Send a query to the Ollama API and return the response."""
    
    # Prepare the request payload
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    # Add system prompt if provided
    if system_prompt:
        payload["system"] = system_prompt
    
    try:
        # Make the API request
        response = requests.post(f"{OLLAMA_API_URL}/api/generate", json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse and return the response
        result = response.json()
        return result.get("response", "No response received")
    
    except requests.exceptions.RequestException as e:
        return f"Error connecting to Ollama API: {str(e)}"

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# App title and description
st.title("🤖 Ollama Chatbot")
st.markdown("Chat with your Ollama models through ngrok")

# Sidebar for model selection and system prompt
with st.sidebar:
    st.header("Settings")
    
    # Model selection
    model = st.selectbox(
        "Select Model",
        ["mistral", "llama2", "gemma", "phi"],
        index=0
    )
    
    # System prompt input
    system_prompt = st.text_area(
        "System Prompt (optional)",
        "You are a helpful AI assistant.",
        help="This sets the behavior of the AI"
    )
    
    # Display connection status
    st.subheader("Connection Status")
    try:
        response = requests.get(f"{OLLAMA_API_URL}/api/tags")
        if response.status_code == 200:
            st.success(f"✅ Connected to Ollama API")
            # Display available models
            models = response.json().get("models", [])
            if models:
                st.write("Available models:")
                for m in models:
                    st.write(f"- {m.get('name')}")
        else:
            st.error("❌ Failed to connect to Ollama API")
    except:
        st.error("❌ Failed to connect to Ollama API")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask something..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Get full chat history for context
            chat_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
            
            # Query Ollama with the full context
            response = query_ollama(
                prompt=chat_history,
                model=model,
                system_prompt=system_prompt
            )
            
            # Display the response
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Add a button to clear chat history
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()
