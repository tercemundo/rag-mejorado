import streamlit as st
import os
import tempfile
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
    if st.session_state.use_only_model_knowledge:
        system_message = """
        Eres un asistente experto que responde preguntas utilizando tu conocimiento general.
        Responde de manera concisa y útil.
        """
        
        user_message = query
        source_type = "conocimiento general (forzado por configuración)"
    
    elif st.session_state.use_only_documents:
        docs = st.session_state.retriever.get_relevant_documents(query, k=5)
        
        if docs:
            context = "\n\n".join([doc.page_content for doc in docs])
            
            system_message = """
            Eres un asistente experto que responde preguntas basándose EXCLUSIVAMENTE en los documentos proporcionados.
            Responde de manera concisa y basándote solo en la información proporcionada.
            Si la información necesaria no está en los documentos, indícalo claramente y NO uses tu conocimiento general.
            """
            
            user_message = f"""
            Contexto de los documentos:
            {context}

            Mi pregunta es: {query}
            """
            
            source_type = "documentos (forzado por configuración)"
        else:
            system_message = """
            Eres un asistente experto que responde preguntas basándose EXCLUSIVAMENTE en los documentos proporcionados.
            Como no hay documentos relevantes para esta consulta, indica que no tienes información.
            NO uses tu conocimiento general.
            """
            
            user_message = query
            source_type = "documentos (sin resultados relevantes, forzado por configuración)"
    
    else:
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
    st.session_state.retriever = SimpleRetriever()  # Inicializar un retriever vacío

if "llm" not in st.session_state:
    st.session_state.llm = None

if "pdfs_processed" not in st.session_state:
    st.session_state.pdfs_processed = False

if "api_key_validated" not in st.session_state:
    st.session_state.api_key_validated = False

if "relevance_threshold" not in st.session_state:
    st.session_state.relevance_threshold = 0.3

if "use_only_model_knowledge" not in st.session_state:
    st.session_state.use_only_model_knowledge = False

if "use_only_documents" not in st.session_state:
    st.session_state.use_only_documents = False

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
        
        # Columnas para las opciones booleanas
        col1, col2 = st.columns(2)
        
        with col1:
            use_only_model = st.checkbox(
                "Usar solo conocimiento del modelo (ignorar PDFs)",
                value=st.session_state.use_only_model_knowledge,
                help="Si activas esta opción, el sistema ignorará los PDFs y usará solo el conocimiento general del modelo"
            )
        
        with col2:
            use_only_docs = st.checkbox(
                "Usar solo documentos (ignorar conocimiento del modelo)",
                value=st.session_state.use_only_documents,
                help="Si activas esta opción, el sistema usará solo la información de los PDFs"
            )
        
        # Lógica para que las opciones sean mutuamente excluyentes
        if use_only_model and not st.session_state.use_only_model_knowledge:
            st.session_state.use_only_model_knowledge = True
            st.session_state.use_only_documents = False
        elif use_only_docs and not st.session_state.use_only_documents:
            st.session_state.use_only_documents = True
            st.session_state.use_only_model_knowledge = False
        elif not use_only_model and st.session_state.use_only_model_knowledge:
            st.session_state.use_only_model_knowledge = False
        elif not use_only_docs and st.session_state.use_only_documents:
            st.session_state.use_only_documents = False
        
        # Mostrar mensaje si ambas opciones están marcadas (no debería ocurrir por la lógica anterior)
        if st.session_state.use_only_model_knowledge and st.session_state.use_only_documents:
            st.error("No puedes seleccionar ambas opciones a la vez. Se ha desactivado la última selección.")
            if use_only_model:
                st.session_state.use_only_documents = False
            else:
                st.session_state.use_only_model_knowledge = False
        
        # Mostrar configuración de relevancia solo si no se está usando exclusivamente el modelo o los documentos
        if not st.session_state.use_only_model_knowledge and not st.session_state.use_only_documents:
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
        # Mostrar uploader de PDFs solo si no se está usando exclusivamente el modelo
        if not st.session_state.use_only_model_knowledge:
            uploaded_files = st.file_uploader("Sube tus archivos PDF", accept_multiple_files=True, type="pdf")
            
            if uploaded_files and not st.session_state.pdfs_processed:
                with st.spinner("Procesando PDFs..."):
                    try:
                        temp_files = []
                        documents = []
                        for file in uploaded_files:
                            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                            temp_file.write(file.read())
                            temp_files.append(temp_file.name)
                            temp_file.close()

                            # Cargar y procesar cada PDF
                            loader = PyPDFLoader(temp_file.name)
                            documents.extend(loader.load())
                        
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
        else:
            # Si se usa solo el conocimiento del modelo, creamos un retriever vacío
            st.session_state.retriever = SimpleRetriever()
            st.session_state.pdfs_processed = True
            st.info("Modo de solo conocimiento del modelo activado. No se usarán PDFs.")

        # Inicializar el modelo LLM si aún no se ha hecho
        if st.session_state.llm is None and st.session_state.api_key_validated:
            try:
                llm = ChatGroq(
                    api_key=st.session_state.groq_api_key,
                    model_name=st.session_state.model_name
                )
                st.session_state.llm = llm
            except Exception as e:
                st.error(f"Error al inicializar el modelo: {str(e)}")

# Área principal para el chat
if not st.session_state.api_key_validated:
    st.info("Por favor, ingresa tu API key de Groq en la barra lateral para comenzar.")
elif not st.session_state.pdfs_processed:
    if st.session_state.use_only_model_knowledge:
        st.session_state.pdfs_processed = True
        st.session_state.retriever = SimpleRetriever()
    else:
        st.info("Por favor, sube al menos un archivo PDF en la barra lateral para comenzar, o activa la opción de usar solo el conocimiento del modelo.")

if st.session_state.api_key_validated and st.session_state.pdfs_processed:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Pregunta sobre tus PDFs o conocimiento general"):
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
        st.session_state.retriever = SimpleRetriever()  # Reinicia el retriever
        st.session_state.llm = None
        st.session_state.pdfs_processed = False
        st.session_state.api_key_validated = False
        st.session_state.relevance_threshold = 0.3  # Restablecer a valor predeterminado
        st.session_state.use_only_model_knowledge = False  # Restablecer opción de solo modelo
        st.session_state.use_only_documents = False  # Restablecer opción de solo documentos
        st.success("Conversación reiniciada.")
