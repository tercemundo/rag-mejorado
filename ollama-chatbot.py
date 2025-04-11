import streamlit as st
import requests
import json
import time
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ollama-chatbot")

# Set page configuration
st.set_page_config(page_title="Ollama Chatbot", page_icon="🤖", layout="wide")

# Ollama API endpoint
OLLAMA_API_URL = "https://5c3e-34-82-129-29.ngrok-free.app"

# Debug information display
debug_container = st.empty()
debug_info = {"api_calls": 0, "response_times": []}

def query_ollama(prompt, model="mistral", system_prompt=None):
    """Send a query to the Ollama API and return the response."""
    
    start_time = time.time()
    logger.debug(f"Starting API call to Ollama with model: {model}")
    
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
        logger.debug(f"Sending request to {OLLAMA_API_URL}/api/generate")
        response = requests.post(f"{OLLAMA_API_URL}/api/generate", json=payload)
        
        # Update debug info
        elapsed_time = time.time() - start_time
        debug_info["api_calls"] += 1
        debug_info["response_times"].append(elapsed_time)
        
        logger.debug(f"Response received in {elapsed_time:.2f} seconds")
        logger.debug(f"Response status code: {response.status_code}")
        
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse and return the response
        result = response.json()
        return result.get("response", "No response received")
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error connecting to Ollama API: {str(e)}")
        return f"Error connecting to Ollama API: {str(e)}"

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# App title and description
st.title("🤖 Ollama Chatbot")
st.markdown("Chat with your Ollama models through ngrok")

# Debug information
with st.expander("Debug Information", expanded=True):
    st.write(f"API URL: {OLLAMA_API_URL}")
    
    # Connection test
    start_time = time.time()
    try:
        logger.debug("Testing connection to Ollama API")
        response = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=5)
        connection_time = time.time() - start_time
        st.write(f"Connection test: ✅ ({connection_time:.2f}s)")
        st.write(f"Status code: {response.status_code}")
    except Exception as e:
        connection_time = time.time() - start_time
        st.write(f"Connection test: ❌ ({connection_time:.2f}s)")
        st.write(f"Error: {str(e)}")
    
    # Display debug metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("API Calls", debug_info["api_calls"])
    with col2:
        avg_time = sum(debug_info["response_times"]) / max(len(debug_info["response_times"]), 1)
        st.metric("Avg Response Time", f"{avg_time:.2f}s")
    
    # System info
    st.write("### System Information")
    import platform
    st.write(f"Python version: {platform.python_version()}")
    st.write(f"Platform: {platform.platform()}")

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
        logger.debug("Fetching available models")
        start_time = time.time()
        response = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=5)
        fetch_time = time.time() - start_time
        
        if response.status_code == 200:
            st.success(f"✅ Connected to Ollama API ({fetch_time:.2f}s)")
            # Display available models
            models = response.json().get("models", [])
            if models:
                st.write("Available models:")
                for m in models:
                    st.write(f"- {m.get('name')}")
            else:
                st.warning("No models found")
        else:
            st.error(f"❌ Failed to connect to Ollama API: Status {response.status_code}")
    except requests.exceptions.Timeout:
        st.error(f"❌ Connection to Ollama API timed out after 5 seconds")
    except Exception as e:
        st.error(f"❌ Failed to connect to Ollama API: {str(e)}")

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
            logger.debug("Processing user input")
            
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

# Footer with debug information
st.markdown("---")
st.markdown(f"**Debug Stats:** API Calls: {debug_info['api_calls']} | Last Response Time: {debug_info['response_times'][-1] if debug_info['response_times'] else 0:.2f}s")
