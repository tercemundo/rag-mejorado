import streamlit as st
from groq import Groq
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
import requests
from bs4 import BeautifulSoup
import uuid

# Set page configuration
st.set_page_config(
    page_title="Groq Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_pdfs" not in st.session_state:
    st.session_state.uploaded_pdfs = {}

# App title and description
st.title("🤖 Groq Chatbot")
st.markdown("Chat with Llama and DeepSeek models using Groq API and your PDF documents")

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

# Function to process uploaded PDF
def process_pdf(uploaded_file):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        # Write the uploaded file data to the temporary file
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Load PDF
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        # Add file name to metadata
        for doc in documents:
            doc.metadata["source"] = uploaded_file.name
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        document_chunks = text_splitter.split_documents(documents)
        
        # Store PDF info
        pdf_id = str(uuid.uuid4())
        st.session_state.uploaded_pdfs[pdf_id] = {
            "name": uploaded_file.name,
            "chunks": document_chunks
        }
        
        # Update all document chunks
        all_chunks = []
        for pdf_info in st.session_state.uploaded_pdfs.values():
            all_chunks.extend(pdf_info["chunks"])
        
        st.session_state.document_chunks = all_chunks
        
        # Create embeddings and vectorstore
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        vectorstore = FAISS.from_documents(all_chunks, embeddings)
        st.session_state.vectorstore = vectorstore
        
        return len(document_chunks), uploaded_file.name
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return 0, uploaded_file.name
    finally:
        # Clean up the temporary file
        os.unlink(tmp_path)

# Function to search the web
def search_web(query, num_results=3):
    try:
        # Use DuckDuckGo search
        search_url = f"https://lite.duckduckgo.com/lite/?q={query}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract search results
        results = []
        for a_tag in soup.find_all('a', href=True):
            if a_tag['href'].startswith('https://') and not a_tag['href'].startswith('https://duckduckgo.com'):
                results.append({
                    "title": a_tag.get_text().strip(),
                    "url": a_tag['href']
                })
                if len(results) >= num_results:
                    break
        
        # Get content from top result
        if results:
            try:
                content_response = requests.get(results[0]["url"], headers=headers, timeout=5)
                content_soup = BeautifulSoup(content_response.text, 'html.parser')
                
                # Extract text from paragraphs
                paragraphs = content_soup.find_all('p')
                content = "\n".join([p.get_text().strip() for p in paragraphs[:5]])
                
                return {
                    "success": True,
                    "results": results,
                    "content": content
                }
            except:
                return {
                    "success": True,
                    "results": results,
                    "content": "Could not retrieve content from the webpage."
                }
        else:
            return {
                "success": False,
                "message": "No search results found."
            }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error searching the web: {str(e)}"
        }

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
    
    # PDF upload section
    st.header("Upload Documents")
    uploaded_file = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    if uploaded_file:
        if st.button("Process PDFs"):
            with st.spinner("Processing PDFs..."):
                total_chunks = 0
                for pdf in uploaded_file:
                    chunks, filename = process_pdf(pdf)
                    if chunks > 0:
                        total_chunks += chunks
                        st.success(f"✅ {filename}: {chunks} chunks")
                    else:
                        st.error(f"❌ Failed to process {filename}")
                
                if total_chunks > 0:
                    st.success(f"All PDFs processed! Total chunks: {total_chunks}")
    
    # Display uploaded PDFs
    if st.session_state.uploaded_pdfs:
        st.subheader("Uploaded PDFs")
        for pdf_id, pdf_info in st.session_state.uploaded_pdfs.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"📄 {pdf_info['name']} ({len(pdf_info['chunks'])} chunks)")
            with col2:
                if st.button("Remove", key=f"remove_{pdf_id}"):
                    del st.session_state.uploaded_pdfs[pdf_id]
                    # Update all document chunks
                    all_chunks = []
                    for remaining_pdf in st.session_state.uploaded_pdfs.values():
                        all_chunks.extend(remaining_pdf["chunks"])
                    
                    st.session_state.document_chunks = all_chunks
                    
                    # Recreate vectorstore if there are still documents
                    if all_chunks:
                        embeddings = HuggingFaceEmbeddings(
                            model_name="sentence-transformers/all-MiniLM-L6-v2",
                            model_kwargs={'device': 'cpu'}
                        )
                        st.session_state.vectorstore = FAISS.from_documents(all_chunks, embeddings)
                    else:
                        st.session_state.vectorstore = None
                    
                    st.rerun()
    
    # Search options
    st.header("Search Options")
    enable_web_search = st.checkbox("Enable web search as fallback", value=True)
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()
    
    # Clear all documents button
    if st.button("Clear All Documents"):
        st.session_state.vectorstore = None
        st.session_state.document_chunks = []
        st.session_state.uploaded_pdfs = {}
        st.success("All documents cleared!")
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
            # STEP 1: Try to find answer in PDFs if documents are available
            pdf_answer = None
            pdf_sources = None
            is_uncertain = False
            if st.session_state.vectorstore is not None:
                with st.spinner("Searching in your documents..."):
                    # Initialize LangChain with Groq
                    llm = ChatGroq(
                        groq_api_key=api_key,
                        model_name=model,
                        temperature=temperature
                    )
                    
                    # Create retrieval chain
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 5}
                    )
                    
                    qa_chain = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        retriever=retriever,
                        return_source_documents=True
                    )
                    
                    # Get response from RAG
                    result = qa_chain({"question": prompt, "chat_history": st.session_state.chat_history})
                    pdf_answer = result["answer"]
                    
                    # Save source information for later use even if answer is uncertain
                    if "source_documents" in result and result["source_documents"]:
                        sources = {}
                        for doc in result["source_documents"]:
                            if hasattr(doc, 'metadata'):
                                source = doc.metadata.get('source', 'Unknown')
                                page = doc.metadata.get('page', 'Unknown page')
                                if source not in sources:
                                    sources[source] = set()
                                sources[source].add(f"Page {page}")
                        
                        if sources:
                            source_text = "\n\n**Sources:**\n"
                            for source, pages in sources.items():
                                pages_str = ", ".join(sorted(pages))
                                source_text += f"- {source} ({pages_str})\n"
                            pdf_sources = source_text
                    
                    # Check if the answer is useful (not just "I don't know" or similar)
                    uncertain_phrases = [
                        "i don't know", "i don't have", "i cannot", "i can't", 
                        "no information", "not mentioned", "not specified",
                        "not provided", "no context", "no data"
                    ]
                    
                    is_uncertain = any(phrase in pdf_answer.lower() for phrase in uncertain_phrases)
                    
                    if not is_uncertain:
                        # Update chat history for RAG context
                        st.session_state.chat_history.append((prompt, pdf_answer))
                        
                        # Add sources information
                        if pdf_sources:
                            pdf_answer += pdf_sources
                        
                        full_response = pdf_answer
                        message_placeholder.markdown(full_response)
            
            # STEP 2: If no good answer from PDFs, use the model directly
            model_is_uncertain = False  # Initialize the variable
            if full_response == "":
                with st.spinner("Thinking..."):
                    client = Groq(api_key=api_key)
                    
                    # Create chat completion
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
                    
                    # Check if model indicates it doesn't know
                    uncertain_phrases = [
                        "i don't know", "i don't have", "i cannot", "i can't", 
                        "no information", "not mentioned", "not specified",
                        "not provided", "no context", "no data"
                    ]
                    
                    model_is_uncertain = any(phrase in full_response.lower() for phrase in uncertain_phrases)
                    
                    # If we have PDF sources but the model gave a better answer, add the sources
                    if not model_is_uncertain and pdf_sources and is_uncertain:
                        full_response += pdf_sources
                        message_placeholder.markdown(full_response)
            
            # STEP 3: If model doesn't know and web search is enabled, try web search
            if enable_web_search and (full_response == "" or model_is_uncertain):
                with st.spinner("Searching the web..."):
                    web_results = search_web(prompt)
                    
                    if web_results["success"] and web_results["results"]:
                        # Format web results
                        web_content = web_results.get("content", "")
                        
                        # Use model to generate answer based on web content
                        client = Groq(api_key=api_key)
                        web_prompt = f"""Based on the following information from the web, please answer the question: "{prompt}"
                        
Web content:
{web_content}

Web search results:
"""
                        for i, result in enumerate(web_results["results"]):
                            web_prompt += f"{i+1}. {result['title']} - {result['url']}\n"
                        
                        web_messages = [{"role": "user", "content": web_prompt}]
                        
                        web_completion = client.chat.completions.create(
                            messages=web_messages,
                            model=model,
                            temperature=temperature
                        )
                        
                        web_answer = web_completion.choices[0].message.content
                        
                        # Add sources from web
                        web_answer += "\n\n**Web Sources:**\n"
                        for i, result in enumerate(web_results["results"][:3]):
                            web_answer += f"{i+1}. [{result['title']}]({result['url']})\n"
                        
                        full_response = web_answer
                        message_placeholder.markdown(full_response)
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            full_response = f"Sorry, an error occurred: {str(e)}"
            message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
