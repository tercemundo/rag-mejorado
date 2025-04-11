import streamlit as st
import requests
import time

# Set page configuration
st.set_page_config(page_title="Ollama Chatbot", page_icon="🤖")

# Ollama API endpoint - Remove trailing slash
OLLAMA_API_URL = "https://d414-34-16-137-7.ngrok-free.app"

# Add a status indicator at the top
status_container = st.empty()
status_container.info("Initializing chatbot...")

# Function to check API connection with timeout
def check_api_connection():
    try:
        start_time = time.time()
        # Print the full URL for debugging
        full_url = f"{OLLAMA_API_URL}/api/tags"
        st.write(f"Connecting to: {full_url}")
        
        response = requests.get(full_url, timeout=3)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            models = response.json().get("models", [])
            if models:
                status_container.success(f"✅ Connected to Ollama API in {elapsed:.2f}s")
                return True, models
            else:
                status_container.warning("⚠️ Connected to Ollama API but no models found")
                st.info("You need to pull the Mistral model first. Run 'ollama pull mistral' on the server.")
                return False, []
        else:
            status_container.error(f"❌ API returned status code {response.status_code}")
            # Add more debug info
            st.error(f"Response content: {response.text[:200]}...")
            st.info("Make sure to run 'ollama pull mistral' on the server before connecting.")
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

# Function to pull model via API (if server allows it)
def pull_model(model_name="mistral"):
    try:
        st.info(f"Attempting to pull model '{model_name}' via API...")
        
        payload = {
            "name": model_name
        }
        
        with st.spinner(f"Pulling {model_name} model (this may take several minutes)..."):
            response = requests.post(
                f"{OLLAMA_API_URL}/api/pull",
                json=payload,
                timeout=300  # 5 minute timeout for model pulling
            )
        
        if response.status_code == 200:
            st.success(f"✅ Successfully pulled {model_name} model!")
            return True
        else:
            st.error(f"Failed to pull model: Status {response.status_code}")
            return False
    except Exception as e:
        st.error(f"Error pulling model: {str(e)}")
        return False

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
                timeout=30
            )
            
        if response.status_code == 200:
            return response.json().get("response", "No response received")
        else:
            return f"Error: API returned status code {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

# App title
st.title("🤖 Ollama Chatbot")

# Add model pulling option
with st.expander("Model Management"):
    st.write("If no models are available, you need to pull one first.")
    model_to_pull = st.selectbox(
        "Select model to pull",
        ["mistral", "llama2", "gemma", "phi"],
        index=0
    )
    if st.button("Pull Selected Model"):
        success = pull_model(model_to_pull)
        if success:
            st.rerun()  # Refresh the app after pulling

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
        st.info("Note: You need to pull models before using them.")
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    # Help section
    with st.expander("Troubleshooting"):
        st.markdown("""
        ### Common Issues:
        
        1. **No models available**: Run 'ollama pull mistral' on the server
        2. **Connection errors**: Check if the ngrok URL is correct and active
        3. **Slow responses**: Large models may take time to generate responses
        
        ### Using with LangChain:
        
        ```python
        from langchain.llms import Ollama
        
        # Initialize Ollama with your server URL
        llm = Ollama(
            base_url="https://your-ngrok-url",
            model="mistral"
        )
        
        # Now you can use it with LangChain
        response = llm("Your prompt here")
        ```
        """)

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
    st.error("Cannot connect to Ollama API or no models available.")
    st.info("Make sure to run 'ollama pull mistral' on the server before connecting.")
