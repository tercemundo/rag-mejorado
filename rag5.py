import streamlit as st
import os
import tempfile
import numpy as np
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

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

# Resto del código similar al anterior, con modificaciones en la lógica de procesamiento de PDFs

# En la sección de sidebar, modificar el uploader de PDFs
if st.session_state.api_key_validated:
    if not st.session_state.use_only_model_knowledge:
        st.subheader("Carga de Documentos")
        
        # Agregar campo de nombre para cada colección de documentos
        document_collections = st.session_state.get('document_collections', {})
        
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
                # Crear retriever multidocumento
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
                        
                        # Añadir colección al retriever
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

# Modificar la función generate_response para manejar múltiples documentos
def generate_response(query):
    # Lógica similar al código anterior, pero ahora manejando múltiples colecciones
    if st.session_state.use_only_model_knowledge:
        # Igual que antes
        pass
    else:
        # Recuperar documentos relevantes de múltiples colecciones
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
            Eres un asistente experto que responde preguntas basándose en documentos de: {sources_text}.
            Responde de manera concisa y basándote solo en la información proporcionada.
            Si la información necesaria no está en los documentos, indícalo claramente.
            """
            
            user_message = f"""
            Contexto de los documentos:
            {context}

            Mi pregunta es: {query}
            """
            
            source_type = f"documentos ({sources_text})"
        else:
            # Si no hay documentos relevantes, usar conocimiento general
            system_message = """
            Eres un asistente experto que responde preguntas utilizando tu conocimiento general.
            Responde de manera concisa y útil.
            """
            
            user_message = query
            source_type = "conocimiento general"
    
    # El resto de la función genera la respuesta como antes
    response = st.session_state.llm.invoke(
        [{"role": "system", "content": system_message},
         {"role": "user", "content": user_message}]
    )
    
    full_response = f"{response.content}\n\n*Respuesta basada en: {source_type}*"
    
    return full_response
