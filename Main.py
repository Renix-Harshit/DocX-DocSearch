import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import PyPDF2
import docx
import pandas as pd
from PIL import Image
import pytesseract
import re
from pdf2image import convert_from_path
import faiss
import numpy as np
from typing import List, Dict
import torch
from sentence_transformers import SentenceTransformer

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class DocumentProcessor:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunk_size = 3  # Number of sentences per chunk

    @staticmethod
    def extract_text(uploaded_file):
        file_type = uploaded_file.type

        try:
            if file_type == "application/pdf":
                try:
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    return " ".join([page.extract_text() for page in pdf_reader.pages])
                except:
                    images = convert_from_path(uploaded_file)
                    return " ".join([pytesseract.image_to_string(img) for img in images])

            elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc = docx.Document(uploaded_file)
                return " ".join([para.text for para in doc.paragraphs])

            elif file_type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
                df = pd.read_excel(uploaded_file)
                return " ".join(df.astype(str).apply(lambda x: ' '.join(x), axis=1))

            elif file_type in ["image/png", "image/jpeg", "image/jpg"]:
                image = Image.open(uploaded_file)
                return pytesseract.image_to_string(image)

            return "Unsupported file type"

        except Exception as e:
            return f"Error extracting text: {str(e)}"

    def preprocess_text(self, text: str) -> List[str]:
        """Split text into chunks for vectorization."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        
        for i in range(0, len(sentences), self.chunk_size):
            chunk = ' '.join(sentences[i:i + self.chunk_size])
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk)
        
        return chunks

    def create_vector_store(self, documents: List[str]) -> tuple:
        """Create FAISS index from documents."""
        all_chunks = []
        chunk_to_doc_map = []  # Maps chunk index to document index
        
        for doc_idx, doc in enumerate(documents):
            chunks = self.preprocess_text(doc)
            all_chunks.extend(chunks)
            chunk_to_doc_map.extend([doc_idx] * len(chunks))

        # Generate embeddings
        embeddings = self.embedder.encode(all_chunks)
        
        # Initialize FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        
        # Add vectors to the index
        index.add(np.array(embeddings).astype('float32'))

        return index, all_chunks, chunk_to_doc_map

    def semantic_search(self, query: str, index, chunks: List[str], chunk_to_doc_map: List[int], k: int = 3) -> List[Dict]:
        """Perform semantic search using FAISS."""
        query_vector = self.embedder.encode([query])
        D, I = index.search(np.array(query_vector).astype('float32'), k)
        
        results = []
        seen_docs = set()
        
        for idx in I[0]:
            doc_idx = chunk_to_doc_map[idx]
            if doc_idx not in seen_docs:  # Avoid duplicate documents 
                seen_docs.add(doc_idx)
                results.append({
                    'document_index': doc_idx,
                    'relevant_passages': [chunks[idx]]
                })
        
        return results

    @staticmethod
    def generate_suggestions(text):
        groq_api_key = os.getenv('GROQ_API_KEY', 'dummy_key')
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

        try:
            prompt = f"""
            Generate 5 unique and insightful questions based on the following text. 
            These questions should cover different aspects and encourage deeper exploration:

            Text: {text[:1000]}

            Format your response as a numbered list of questions.
            """
            response = llm.invoke(prompt)
            suggestions = [
                line.strip().split('. ', 1)[-1].strip()
                for line in response.content.split('\n')
                if line.strip() and line.strip()[0].isdigit()
            ]
            return suggestions[:5]
        except Exception as e:
            st.error(f"Error generating suggestions: {e}")
            return []

def main():
    st.title("📄 Advanced Document Search System with FAISS")

    # Initialize session state
    if 'documents' not in st.session_state:
        st.session_state.documents = []
        st.session_state.file_details = []
        st.session_state.all_suggestions = []
        st.session_state.search_results = None
        st.session_state.search_query = ""
        st.session_state.vector_store = None
        st.session_state.chunks = None
        st.session_state.chunk_map = None
        st.session_state.doc_processor = DocumentProcessor()

    uploaded_files = st.file_uploader(
        "Upload up to 3 Documents (PDF, DOCX, XLSX, PNG, JPG)", 
        type=["pdf", "docx", "xlsx", "png", "jpg"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        uploaded_files = uploaded_files[:3]
        
        with st.spinner("Processing documents and creating vector store..."):
            st.session_state.documents.clear()
            st.session_state.file_details.clear()
            st.session_state.all_suggestions.clear()

            for file in uploaded_files:
                doc_text = DocumentProcessor.extract_text(file)
                st.session_state.documents.append(doc_text)
                st.session_state.file_details.append({
                    'name': file.name,
                    'file': file
                })
                suggestions = DocumentProcessor.generate_suggestions(doc_text)
                st.session_state.all_suggestions.extend(suggestions)

            #V1-vector store
            st.session_state.vector_store, st.session_state.chunks, st.session_state.chunk_map = (
                st.session_state.doc_processor.create_vector_store(st.session_state.documents)
            )

    if st.session_state.documents:
        st.header("📋 Document Suggestions")
        suggestion_options = ["Select a suggestion..."] + st.session_state.all_suggestions

        def suggestion_selected():
            """Callback for when a suggestion is selected from the dropdown."""
            print(f"Selected Suggestion: {st.session_state.selected_suggestion}")  # DebuG 1 print statement
            if st.session_state.selected_suggestion != "Select a suggestion...":
                st.session_state.search_query = st.session_state.selected_suggestion
                st.session_state.search_results = st.session_state.doc_processor.semantic_search(
                    st.session_state.search_query,
                    st.session_state.vector_store,
                    st.session_state.chunks,
                    st.session_state.chunk_map
                )
                print(f"Search results: {st.session_state.search_results}")  # Debug2 print statement

        selected_suggestion = st.selectbox(
            "Quick Search Suggestions",
            options=suggestion_options,
            key="selected_suggestion",
            on_change=suggestion_selected
        )

        st.header("🔍 Semantic Search")
        search_query = st.text_input(
            "Enter your search query:", 
            value=st.session_state.search_query,
            key="search_input",
            on_change=lambda: setattr(st.session_state, 'search_results',
                st.session_state.doc_processor.semantic_search(
                    st.session_state.search_query,
                    st.session_state.vector_store,
                    st.session_state.chunks,
                    st.session_state.chunk_map
                ))
        )
        st.session_state.search_query = search_query

        if st.button("Search"):
            st.session_state.search_results = st.session_state.doc_processor.semantic_search(
                st.session_state.search_query,
                st.session_state.vector_store,
                st.session_state.chunks,
                st.session_state.chunk_map
            )

        st.header("Search Results")
        if st.session_state.search_results:
            for result in st.session_state.search_results:
                doc_name = st.session_state.file_details[result['document_index']]['name']
                st.write(f"**Found relevant information in {doc_name}:**")
                for passage in result['relevant_passages']:
                    st.write(f"- {passage}")
                st.download_button(
                    label=f"Download {doc_name}", 
                    data=st.session_state.file_details[result['document_index']]['file'].getvalue(),
                    file_name=doc_name
                )
        else:
            st.write("No search results to display.")
    else:
        st.warning("No documents uploaded yet.")

    with st.sidebar:
        st.title("📚 Document Search Features")
        st.markdown("""
        ## Capabilities:
        - Upload PDF, DOCX, XLSX, PNG, JPG
        - Automatic text extraction
        - FAISS vector database for semantic search
        - AI-powered suggestions
        - Direct document download
        """)
        st.write("Made with ❤️ by Harshit")

if __name__ == "__main__":
    main()