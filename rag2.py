import streamlit as st
import os
import tempfile
import numpy as np
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

# Configurar página de Streamlit
st.set_page_config(page_title="Chat con tus PDFs usando Groq", layout="wide")
st.title("Chat con tus PDFs usando Groq")

# Clase simple para almacenamiento y recuperación de texto
class SimpleRetriever:
    def __init__(self):
        self.documents = []
    
    def add_documents(self, documents):
        self.documents.extend(documents)
    
    def get_relevant_documents(self, query, k=3):
        query_words = set(query.lower().split())
        scored_docs = []
        
        for doc in self.documents:
            content = doc.page_content.lower()
            word_matches = sum(1 for word in query_words if word in content)
            score = word_matches / len(query_words) if query_words else 0
            scored_docs.append((doc, score))
        
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:k]]

# Función para evaluar la relevancia de los documentos para la consulta
def evaluate_document_relevance(docs, query):
    if not docs:
        return 0.0
    
    query_words = set(query.lower().split())
    if not query_words:
        return 0.0
    
    total_relevance = 0
    for doc in docs:
        content = doc.page_content.lower()
        word_matches = sum(1 for word in query_words if word in content)
        doc_relevance = word_matches / len(query_words)
        total_relevance += doc_relevance
    
    return total_relevance / len(docs)

# Función para generar respuesta
def generate_response(query):
    docs = st.session_state.retriever.get_relevant_documents(query, k=5)
    relevance_score = evaluate_document_relevance(docs, query)
    
    if relevance_score >= st.session_state.relevance_threshold and docs:
        context = "\n\n".join([doc.page_content for doc in docs])
        
        system_message = """
        Eres un asistente experto que responde preguntas basándose en los documentos proporcionados.
        Responde de manera concisa y basándote solo en la información proporcionada.
        Si la información necesaria no está en los documentos, indícalo claramente.
        """
        
        user_message = f"""
        Contexto de los documentos:
        {context}

        Mi pregunta es: {query}
        """
        
        source_type = "documentos"
    else:
        system_message = """
        Eres un asistente experto que responde preguntas utilizando tu conocimiento general.
        Responde de manera concisa y útil.
        """
        
        user_message = query
        source_type = "conocimiento general"
    
    response = st.session_state.llm.invoke(
        [{"role": "system", "content": system_message},
         {"role": "user", "content": user_message}]
    )
    
    full_response = f"{response.content}\n\n*Respuesta basada en: {source_type}*"
    
    return full_response

# Inicializar variables de estado en la sesión
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "llm" not in st.session_state:
    st.session_state.llm = None

if "pdfs_processed" not in st.session_state:
    st.session_state.pdfs_processed = False

if "api_key_validated" not in st.session_state:
    st.session_state.api_key_validated = False

if "relevance_threshold" not in st.session_state:
    st.session_state.relevance_threshold = 0.3

# Cargar variables de entorno
load_dotenv()

# Sidebar para configuración
with st.sidebar:
    st.header("Configuración")
    
    api_key_input = st.text_input(
        "API Key de Groq", 
        value=os.environ.get("GROQ_API_KEY", ""),
        type="password",
        help="Ingresa tu API key de Groq. Esta no será almacenada permanentemente."
    )
    
    if api_key_input:
        st.session_state.groq_api_key = api_key_input
        st.session_state.api_key_validated = True
        st.success("API key configurada ✅")
    else:
        st.warning("Por favor, ingresa tu API key de Groq para continuar")
        
    if st.session_state.api_key_validated:
        model_name = st.selectbox(
            "Selecciona modelo de Groq",
            ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"]
        )
        
        st.session_state.model_name = model_name
        
        relevance_threshold = st.slider(
            "Umbral de relevancia para documentos (0-1)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.relevance_threshold,
            step=0.05,
            help="Si la relevancia de los documentos está por debajo de este umbral, el sistema usará el conocimiento general del modelo"
        )
        
        st.session_state.relevance_threshold = relevance_threshold
    
    if st.session_state.api_key_validated:
        uploaded_files = st.file_uploader("Sube tus archivos PDF", accept_multiple_files=True, type="pdf")
        
        if uploaded_files and not st.session_state.pdfs_processed:
            with st.spinner("Procesando PDFs..."):
                try:
                    temp_files = []
                    for file in uploaded_files:
                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                        temp_file.write(file.read())
                        temp_files.append(temp_file.name)
                        temp_file.close()
                    
                    documents = []
                    for path in temp_files:
                        try:
                            loader = PyPDFLoader(path)
                            documents.extend(loader.load())
                        except Exception as e:
                            st.error(f"Error al cargar {path}: {e}")
                    
                    if not documents:
                        st.error("No se pudieron cargar documentos.")
                        st.stop()
                    
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, 
                        chunk_overlap=100
                    )
                    chunks = text_splitter.split_documents(documents)
                    
                    retriever = SimpleRetriever()
                    retriever.add_documents(chunks)
                    st.session_state.retriever = retriever
                    
                    llm = ChatGroq(
                        api_key=st.session_state.groq_api_key,
                        model_name=st.session_state.model_name
                    )
                    st.session_state.llm = llm
                    
                    for path in temp_files:
                        try:
                            os.unlink(path)
                        except:
                            pass
                    
                    st.session_state.pdfs_processed = True
                    st.success(f"Se procesaron {len(chunks)} fragmentos de texto de {len(uploaded_files)} PDFs")
                
                except Exception as e:
                    st.error(f"Error al procesar los documentos: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

# Área principal para el chat
if not st.session_state.api_key_validated:
    st.info("Por favor, ingresa tu API key de Groq en la barra lateral para comenzar.")
elif not st.session_state.pdfs_processed:
    st.info("Por favor, sube al menos un archivo PDF en la barra lateral para comenzar.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Pregunta sobre tus PDFs"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            try:
                with st.spinner("Pensando..."):
                    response = generate_response(prompt)
                    st.markdown(response)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"Error al generar respuesta: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Botón para reiniciar la conversación
if st.session_state.pdfs_processed and st.session_state.messages:
    if st.button("Reiniciar conversación"):
        st.session_state.messages = []
        st.experimental_rerun()
