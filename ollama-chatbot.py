import streamlit as st
import requests
import time

# Set page configuration
st.set_page_config(page_title="Ollama Chatbot", page_icon="🤖")

# Ollama API endpoint
OLLAMA_API_URL = "https://5c3e-34-82-129-29.ngrok-free.app"

# Add a status indicator at the top
status_container = st.empty()
status_container.info("Initializing chatbot...")

# Function to check API connection with timeout
def check_api_connection():
    try:
        start_time = time.time()
        response = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=3)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            status_container.success(f"✅ Connected to Ollama API in {elapsed:.2f}s")
            return True, response.json().get("models", [])
        else:
            status_container.error(f"❌ API returned status code {response.status_code}")
            return False, []
    except requests.exceptions.Timeout:
        status_container.error("❌ Connection timed out after 3 seconds")
        return False, []
    except requests.exceptions.ConnectionError:
        status_container.error("❌ Failed to connect to API")
        return False, []
    except Exception as e:
        status_container.error(f"❌ Error: {str(e)}")
        return False, []

# Simple function to query Ollama
def query_ollama(prompt, model="mistral"):
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        with st.spinner("Generating response..."):
            response = requests.post(
                f"{OLLAMA_API_URL}/api/generate", 
                json=payload,
                timeout=10
            )
            
        if response.status_code == 200:
            return response.json().get("response", "No response received")
        else:
            return f"Error: API returned status code {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

# App title
st.title("🤖 Ollama Chatbot")

# Check connection at startup
connected, available_models = check_api_connection()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar with minimal options
with st.sidebar:
    st.header("Settings")
    
    # Model selection (only if connected)
    if connected and available_models:
        model_names = [m.get("name") for m in available_models]
        model = st.selectbox("Select Model", model_names, index=0)
    else:
        model = "mistral"  # Default model
        st.warning("Using default model: mistral")
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input (only enable if connected)
if connected:
    user_input = st.chat_input("Ask something...")
    if user_input:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Get and display response
        with st.chat_message("assistant"):
            response = query_ollama(user_input, model)
            st.write(response)
        
        # Add assistant response to chat
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.error("Cannot connect to Ollama API. Please check the connection and reload.")
