import streamlit as st
from groq import Groq
import os

# Set page configuration
st.set_page_config(
    page_title="Groq Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# App title and description
st.title("🤖 Groq Chatbot")
st.markdown("Chat with Llama and DeepSeek models using Groq API")

# Function to get API key from secrets or user input
def get_api_key():
    # Try to get API key from Streamlit secrets
    try:
        return st.secrets["GROQ_API_KEY"]
    except:
        # If not in secrets, get from session state or user input
        if "api_key" not in st.session_state or not st.session_state.api_key:
            st.session_state.api_key = ""
        
        api_key = st.sidebar.text_input("Enter Groq API Key:", type="password", value=st.session_state.api_key)
        if api_key:
            st.session_state.api_key = api_key
        return api_key

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Get API key
    api_key = get_api_key()
    
    # Model selection
    model = st.selectbox(
        "Select Model:",
        [
            "llama3-8b-8192",
            "llama3-70b-8192",
            "llama2-70b-4096",
            "deepseek-coder-33b-instruct"
        ]
    )
    
    # Temperature slider
    temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask something..."):
    # Check if API key is provided
    if not api_key:
        st.error("Please enter your Groq API key in the sidebar or configure it in Streamlit secrets.")
        st.stop()
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response with a spinner
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Initialize Groq client
            client = Groq(api_key=api_key)
            
            # Create chat completion
            with st.spinner("Thinking..."):
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": m["role"], "content": m["content"]} 
                        for m in st.session_state.messages
                    ],
                    model=model,
                    temperature=temperature,
                    stream=True
                )
                
                # Stream the response
                for chunk in chat_completion:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            full_response = f"Sorry, an error occurred: {str(e)}"
            message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
