import streamlit as st
import os
import tempfile
import numpy as np
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

# Inicializar variables de estado si no existen
if 'api_key_validated' not in st.session_state:
    st.session_state.api_key_validated = False
if 'groq_api_key' not in st.session_state:
    st.session_state.groq_api_key = ""
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'pdfs_processed' not in st.session_state:
    st.session_state.pdfs_processed = False
if 'document_collections' not in st.session_state:
    st.session_state.document_collections = {}
if 'relevance_threshold' not in st.session_state:
    st.session_state.relevance_threshold = 0.3
if 'use_only_model_knowledge' not in st.session_state:
    st.session_state.use_only_model_knowledge = False
if 'use_only_documents' not in st.session_state:
    st.session_state.use_only_documents = False
if 'model_name' not in st.session_state:
    st.session_state.model_name = "llama3-8b-8192"

# Configurar página de Streamlit
st.set_page_config(page_title="Chat con PDFs usando Groq", layout="wide")
st.title("Chat Multicontext con PDFs usando Groq")

# Clase de Retriever Mejorada para Múltiples Documentos
class MultiDocumentRetriever:
    def __init__(self):
        self.document_collections = {}
    
    def add_document_collection(self, collection_name, documents):
        """Añade una colección de documentos con un nombre específico"""
        self.document_collections[collection_name] = documents
    
    def get_relevant_documents(self, query, k=3):
        """Recupera documentos relevantes de todas las colecciones"""
        query_words = set(query.lower().split())
        scored_docs = []
        
        for collection_name, documents in self.document_collections.items():
            for doc in documents:
                content = doc.page_content.lower()
                word_matches = sum(1 for word in query_words if word in content)
                score = word_matches / len(query_words) if query_words else 0
                scored_docs.append((doc, score, collection_name))
        
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [(doc, collection) for doc, score, collection in scored_docs[:k]]

# Función para validar API Key
def validate_groq_api_key(api_key):
    try:
        # Intentamos crear un cliente de Groq para validar la API key
        test_llm = ChatGroq(
            api_key=api_key,
            model_name="llama3-8b-8192"
        )
        # Prueba simple de invocación
        test_response = test_llm.invoke([
            {"role": "system", "content": "Eres un asistente de prueba."},
            {"role": "user", "content": "Hola"}
        ])
        return True
    except Exception as e:
        st.error(f"Error al validar API Key: {str(e)}")
        return False

# Función para generar respuesta
def generate_response(query):
    # Verificar si se está usando solo conocimiento del modelo
    if st.session_state.use_only_model_knowledge:
        system_message = """
        Eres un asistente experto que responde preguntas utilizando tu conocimiento general.
        Responde de manera concisa y útil.
        """
        
        user_message = query
        source_type = "conocimiento general (forzado por configuración)"
    
    # Verificar si se está usando solo documentos
    elif st.session_state.use_only_documents:
        docs_with_sources = st.session_state.retriever.get_relevant_documents(query, k=5)
        
        if docs_with_sources:
            context_parts = []
            sources_used = set()
            
            for doc, source in docs_with_sources:
                context_parts.append(f"[{source}]: {doc.page_content}")
                sources_used.add(source)
            
            context = "\n\n".join(context_parts)
            sources_text = ", ".join(sources_used)
            
            system_message = f"""
            Eres un asistente experto que responde preguntas basándote EXCLUSIVAMENTE en los documentos de: {sources_text}.
            Responde de manera concisa y basándote solo en la información proporcionada.
            Si la información necesaria no está en los documentos, indícalo claramente.
            """
            
            user_message = f"""
            Contexto de los documentos:
            {context}

            Mi pregunta es: {query}
            """
            
            source_type = f"documentos ({sources_text}, forzado por configuración)"
        else:
            system_message = """
            Eres un asistente experto que responde preguntas basándote EXCLUSIVAMENTE en los documentos proporcionados.
            Como no hay documentos relevantes, indica que no tienes información.
            """
            
            user_message = query
            source_type = "documentos (sin resultados relevantes)"
    
    # Modo híbrido (predeterminado)
    else:
        docs_with_sources = st.session_state.retriever.get_relevant_documents(query, k=5)
        
        if docs_with_sources:
            context_parts = []
            sources_used = set()
            
            for doc, source in docs_with_sources:
                context_parts.append(f"[{source}]: {doc.page_content}")
                sources_used.add(source)
            
            context = "\n\n".join(context_parts)
            sources_text = ", ".join(sources_used)
            
            system_message = f"""
            Eres un asistente experto que responde preguntas basándote en documentos de: {sources_text}.
            Responde de manera concisa y basándote solo en la información proporcionada.
            Si la información necesaria no está en los documentos, puedes usar tu conocimiento general.
            """
            
            user_message = f"""
            Contexto de los documentos:
            {context}

            Mi pregunta es: {query}
            """
            
            source_type = f"documentos ({sources_text})"
        else:
            system_message = """
            Eres un asistente experto que responde preguntas utilizando tu conocimiento general.
            Responde de manera concisa y útil.
            """
            
            user_message = query
            source_type = "conocimiento general"
    
    # Generar respuesta
    response = st.session_state.llm.invoke(
        [{"role": "system", "content": system_message},
         {"role": "user", "content": user_message}]
    )
    
    full_response = f"{response.content}\n\n*Respuesta basada en: {source_type}*"
    
    return full_response

# Función principal de la aplicación
def main():
    # Si la API key no está validada, mostrar pantalla de inicio de sesión
    if not st.session_state.api_key_validated:
        st.header("Configuración Inicial")
        
        # Input para API Key
        api_key_input = st.text_input(
            "Introduce tu API Key de Groq", 
            type="password",
            help="Necesitas una API key de Groq para usar esta aplicación"
        )
        
        # Selección de modelo
        model_name = st.selectbox(
            "Selecciona modelo de Groq",
            ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"]
        )
        
        # Botón de validación
        if st.button("Validar API Key"):
            if api_key_input:
                # Intentar validar la API key
                if validate_groq_api_key(api_key_input):
                    st.session_state.api_key_validated = True
                    st.session_state.groq_api_key = api_key_input
                    st.session_state.model_name = model_name
                    
                    # Inicializar el modelo LLM
                    st.session_state.llm = ChatGroq(
                        api_key=st.session_state.groq_api_key,
                        model_name=st.session_state.model_name
                    )
                    
                    st.success("API Key validada correctamente")
                    st.experimental_rerun()
                else:
                    st.error("La API Key no es válida. Por favor, verifica e intenta nuevamente.")
            else:
                st.warning("Por favor, introduce una API Key")
    
    # Si la API key está validada, mostrar la aplicación principal
    else:
        # Sidebar para configuración de documentos
        with st.sidebar:
            st.header("Configuración")
            
            # Opción de cerrar sesión
            if st.button("Cerrar Sesión"):
                st.session_state.api_key_validated = False
                st.session_state.groq_api_key = ""
                st.session_state.llm = None
                st.experimental_rerun()
            
            # Selección de modelo
            st.session_state.model_name = st.selectbox(
                "Selecciona modelo de Groq",
                ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"],
                index=["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"].index(st.session_state.model_name)
            )
            
            # Opciones de uso de conocimiento
            col1, col2 = st.columns(2)
            
            with col1:
                use_only_model = st.checkbox(
                    "Usar solo conocimiento del modelo",
                    value=st.session_state.use_only_model_knowledge,
                    help="Ignorar PDFs, usar solo conocimiento general"
                )
            
            with col2:
                use_only_docs = st.checkbox(
                    "Usar solo documentos",
                    value=st.session_state.use_only_documents,
                    help="Usar solo información de PDFs"
                )
            
            # Lógica para opciones mutuamente excluyentes
            if use_only_model and not st.session_state.use_only_model_knowledge:
                st.session_state.use_only_model_knowledge = True
                st.session_state.use_only_documents = False
            elif use_only_docs and not st.session_state.use_only_documents:
                st.session_state.use_only_documents = True
                st.session_state.use_only_model_knowledge = False
            
            # Umbral de relevancia (solo en modo híbrido)
            if not st.session_state.use_only_model_knowledge and not st.session_state.use_only_documents:
                st.session_state.relevance_threshold = st.slider(
                    "Umbral de relevancia para documentos",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.relevance_threshold,
                    step=0.05
                )
            
            # Carga de documentos (si no está en modo solo conocimiento)
            if not st.session_state.use_only_model_knowledge:
                st.subheader("Carga de Documentos")
                
                document_collections = st.session_state.document_collections
                
                for i in range(len(document_collections) + 1):
                    col_name_key = f'collection_name_{i}'
                    col_files_key = f'collection_files_{i}'
                    
                    collection_name = st.text_input(
                        f"Nombre de la colección {i+1}", 
                        key=col_name_key, 
                        placeholder="Ej: DevOps, ObraSocial"
                    )
                    
                    uploaded_files = st.file_uploader(
                        f"PDFs para la colección {i+1}", 
                        accept_multiple_files=True, 
                        type="pdf", 
                        key=col_files_key
                    )
                    
                    if collection_name and uploaded_files:
                        if collection_name not in document_collections:
                            document_collections[collection_name] = uploaded_files
                
                st.session_state.document_collections = document_collections
                
                if st.button("Procesar Documentos"):
                    with st.spinner("Procesando PDFs..."):
                        multi_retriever = MultiDocumentRetriever()
                        
                        for collection_name, files in document_collections.items():
                            try:
                                temp_files = []
                                for file in files:
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
                                
                                text_splitter = RecursiveCharacterTextSplitter(
                                    chunk_size=1000, 
                                    chunk_overlap=100
                                )
                                chunks = text_splitter.split_documents(documents)
                                
                                multi_retriever.add_document_collection(collection_name, chunks)
                                
                                for path in temp_files:
                                    try:
                                        os.unlink(path)
                                    except:
                                        pass
                            
                            except Exception as e:
                                st.error(f"Error al procesar documentos de {collection_name}: {str(e)}")
                        
                        st.session_state.retriever = multi_retriever
                        st.session_state.pdfs_processed = True
                        st.success("Documentos procesados exitosamente")
        
        # Área principal de chat
        if not st.session_state.pdfs_processed and not st.session_state.use_only_model_knowledge:
            st.info("Por favor, carga documentos o selecciona modo de conocimiento general")
        else:
            # Mostrar historial de mensajes
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Input de chat
            if prompt := st.chat_input("Pregunta sobre tus PDFs o conocimiento general"):
                # Añadir mensaje del usuario
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generar respuesta
                with st.chat_message("assistant"):
                    try:
                        with st.spinner("Generando respuesta..."):
                            response = generate_response(prompt)
                            st.markdown(response)
                        
                        # Añadir respuesta al historial
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_msg = f"Error al generar respuesta: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
            # Botón para reiniciar conversación
            if st.button("Reiniciar Conversación"):
                st.session_state.messages = []
                st.session_state.pdfs_processed = False
                st.session_state.document_collections = {}
                st.session_state.retriever = None

# Ejecutar la aplicación
if __name__ == "__main__":
    main()
