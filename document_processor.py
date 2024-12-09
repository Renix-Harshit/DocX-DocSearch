import os
import re
import PyPDF2
import docx
import pandas as pd
import pytesseract
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class DocumentProcessor:
    @staticmethod
    def extract_text(uploaded_file):
        file_type = uploaded_file.type
        try:
            if file_type == "application/pdf":
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                return " ".join([page.extract_text() for page in pdf_reader.pages])
            elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc = docx.Document(uploaded_file)
                return " ".join([para.text for para in doc.paragraphs])
            elif file_type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
                df = pd.read_excel(uploaded_file)
                return " ".join(df.astype(str).apply(lambda x: ' '.join(x), axis=1))
            elif file_type in ["image/png", "image/jpeg"]:
                image = Image.open(uploaded_file)
                return pytesseract.image_to_string(image)
            return ""
        except Exception as e:
            return f"Error extracting text: {str(e)}"

    @staticmethod
    def generate_suggestions(text, num_suggestions=5):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = text.split()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        meaningful_words = [word for word in words if word not in stop_words and len(word) > 3]
        word_freq = {}
        for word in meaningful_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        suggestions = sorted(word_freq, key=word_freq.get, reverse=True)[:num_suggestions]
        return suggestions

    @staticmethod
    def advanced_search(documents, query, top_k=3):
        query = query.lower()
        query = re.sub(r'[^a-zA-Z\s]', '', query)
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
        search_results = []
        for i, doc in enumerate(documents):
            doc_processed = doc.lower()
            doc_processed = re.sub(r'[^a-zA-Z\s]', '', doc_processed)
            if query in doc_processed:
                passages = [p.strip() for p in doc.split('.') if query in p.lower()]
                search_results.append({
                    'document_index': i,
                    'relevant_passages': passages[:3]
                })
        return search_results
