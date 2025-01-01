import os
from langchain_groq import ChatGroq
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
import streamlit as st


# Configuration for Tesseract path
# Use environment variable for flexibility across different systems
TESSERACT_PATH = os.getenv('TESSERACT_PATH', r"C:\Program Files\Tesseract-OCR\tesseract.exe")
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

class DocumentProcessor:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunk_size = 3  # Number of sentences per chunk

    @staticmethod
    def extract_text(uploaded_file):
        file_type = uploaded_file.type
        text = ""  # Initialize text here

        try:
            if file_type == "application/pdf":
                try:
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    text = " ".join([page.extract_text() for page in pdf_reader.pages])
                except Exception as pdf_ex:
                   
                    images = convert_from_path(uploaded_file)
                    text = " ".join([pytesseract.image_to_string(img) for img in images])
            elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc = docx.Document(uploaded_file)
                text = " ".join([para.text for para in doc.paragraphs])

            elif file_type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
                df = pd.read_excel(uploaded_file)
                text = " ".join(df.astype(str).apply(lambda x: ' '.join(x), axis=1))

            elif file_type in ["image/png", "image/jpeg", "image/jpg"]:
                image = Image.open(uploaded_file)
                text = pytesseract.image_to_string(image)
            else:
                 text = "Unsupported file type"

        except Exception as e:
             text = f"Error extracting text: {str(e)}"
        finally:
             return text # Return the text
            
    def preprocess_text(self, text: str) -> List[str]:
        """Split text into chunks for vectorization."""
        # Remove leading/trailing whitespaces and replace multiple spaces with a single space
        text = ' '.join(text.split())
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

    def extract_answer(self, question, context):
      groq_api_key = os.getenv('GROQ_API_KEY', 'dummy_key')
      llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
      try:
          prompt = f"""
              Based on the following context, please provide a direct answer to the question.
              If the answer cannot be found in the context, respond with "I cannot find an answer in the document".
              
              Context: {context}
              
              Question: {question}
              
              Answer:
          """
          response = llm.invoke(prompt)
          return response.content.strip()
      except Exception as e:
           st.error(f"Error extracting answer: {e}")
           return "Error getting answer"

    def semantic_search(self, query: str, index, chunks: List[str], chunk_to_doc_map: List[int], k: int = 3) -> List[Dict]:
        """Perform semantic search using FAISS."""
        query_vector = self.embedder.encode([query])
        D, I = index.search(np.array(query_vector).astype('float32'), k)
        
        results = []
        
        for distance, idx in zip(D[0], I[0]):
            doc_idx = chunk_to_doc_map[idx]
            answer = self.extract_answer(query, chunks[idx])
            results.append({
                'document_index': doc_idx,
                'question' : query,
                'answer': answer,
                'distance': distance
            })
        
        #Sort results by distance (lower distance is better)
        results.sort(key=lambda x: x['distance'])
        
        #Group chunks by document
        grouped_results = {}
        for res in results:
            if res['document_index'] not in grouped_results:
                grouped_results[res['document_index']] = {
                   'questions_answers' : [],
                   'distance' : res['distance']
                   
                }
            grouped_results[res['document_index']]['questions_answers'].append({
              'question': res['question'],
                'answer': res['answer']
                })
        
        # Convert grouped results back to list of dictionaries format
        final_results = []
        for doc_idx, data in grouped_results.items():
            final_results.append({
                'document_index': doc_idx,
                'questions_answers' : data['questions_answers'],
                'distance' : data['distance']
            })
        
        return final_results

    @staticmethod
    def generate_suggestions(text):
        groq_api_key = os.getenv('GROQ_API_KEY', 'dummy_key')
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
        
        try:
            prompt = f"""
                Based on the following text, generate 5 questions that a user might ask, 
                focusing specifically on details typically found in documents such as invoices
                (e.g., invoice numbers, amounts, dates, locations, and parties involved).  Make sure to generate specific examples instead of "What is amount of invoice ?".

                Here are some examples for types of question:

                1. "Who is invoice [specific invoice number, e.g. 23579] billed to?"
                2. "What is the total amount of invoice [specific invoice number]?"
                3. "When was invoice [specific invoice number] created?"
                4. "Where was invoice [specific invoice number] shipped to?"
                5. "What items are listed in invoice [specific invoice number]?"

                 Format your response as a numbered list, only focusing on numbers, names and locations, avoid general questions about document such as (give details of the document)

                 Text: {text[:1000]}
                 """
                 
            response = llm.invoke(prompt)
            # split the content and check if line is not empty, first char is a digit then split to take the second part as suggestion
            suggestions = [
                line.strip().split('. ', 1)[-1].strip()
                for line in response.content.split('\n')
                if line.strip() and line.strip()[0].isdigit()
            ]
            return suggestions[:5]
        except Exception as e:
            st.error(f"Error generating suggestions: {e}")
        return []
