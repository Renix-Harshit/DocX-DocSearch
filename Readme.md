# Docx-DRS: Advanced Document Retrieval System 📄🔍

## 1. Project Overview 🚀
Docx-DRS is an advanced document search and retrieval system built using Python and Streamlit, designed to process and search through multiple document types with intelligent text extraction and search capabilities.

## 2. Technical Architecture 🏗️

### 2.1 System Components 🛠️
The system consists of two primary Python modules:
- **document_processor.py**: Core text extraction and search logic
- **app.py**: Streamlit user interface and application workflow

## 3. DocumentProcessor Class 📝

### 3.1 Text Extraction Method 🧠
The `extract_text()` method supports multiple file formats:
- **PDF**: Uses `PyPDF2` for text extraction 📚
- **DOCX**: Utilizes `python-docx` to extract paragraph text 📄
- **XLSX**: Converts Excel data to string representation 📊
- **PNG/JPEG**: Leverages `Tesseract OCR` for image text recognition 🖼️

### 3.2 Suggestion Generation 💡
The `generate_suggestions()` method creates meaningful search suggestions by:
- Converting text to lowercase 🔠
- Removing non-alphabetic characters ✖️
- Filtering out common stop words 🚫
- Identifying top 5 most frequent meaningful words 🏅

### 3.3 Advanced Search Mechanism 🔍
The `advanced_search()` method provides intelligent document searching through:
- Case-insensitive search functionality 🔠🔍
- Removal of non-alphabetic characters ✂️
- Extraction of relevant text passages 📜
- TF-IDF based search algorithm 🧮

## 4. Streamlit User Interface 🌐

### 4.1 Key Features ✨
- Upload up to 3 documents simultaneously 📤
- Support for PDF, DOCX, XLSX, PNG, JPG formats 📄📊🖼️
- Intelligent document suggestions 💡
- Flexible search across documents 🔄
- Direct document download functionality ⬇️

## 5. Technical Dependencies ⚙️
- **Streamlit** 🌐
- **PyPDF2** 📚
- **python-docx** 📄
- **pandas** 📊
- **pytesseract** 🖼️
- **Pillow (PIL)** 🖼️
- **scikit-learn** 📚

## 6. System Limitations ⚠️
- Limited to 3 simultaneous document uploads 📂
- Basic OCR capabilities 🖼️
- Simple TF-IDF search algorithm 🧮

## 7. Future Enhancement Roadmap 📅
- Advanced NLP search techniques 🧠
- Expanded file type support 🗂️
- Improved OCR accuracy 🔍
- Machine learning-based document similarity 🤖

## 8. Conclusion 🎉
Docx-DRS represents an innovative approach to document retrieval, combining multiple technologies to provide a user-friendly, versatile document search experience.

## 9. About 💬
Created with ❤️ by Renix

---

## Installation Process 🛠️

### 1. Clone the Repository

Start by cloning the repository to your local machine:

```bash
git clone https://github.com/your-username/Docx-DocSearch.git
cd Docx-DocSearch
```

### 2. Set Up a Virtual Environment (Optional but Recommended)

Create a virtual environment to isolate the project dependencies:

For Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

For macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install the required Python dependencies using pip:

```bash
pip install -r requirements.txt
```

This will install all necessary libraries like Streamlit, PyPDF2, python-docx, pytesseract, and others.

### 4. Install Tesseract (for OCR)

Since the project uses Tesseract OCR for extracting text from images, make sure Tesseract is installed on your system.

- **Windows**: Download the installer from Tesseract Official and follow the installation instructions.
- **macOS**: Install via Homebrew:
  ```bash
  brew install tesseract
  ```
- **Linux**: Use your package manager to install Tesseract (for example, on Ubuntu):
  ```bash
  sudo apt-get install tesseract-ocr
  ```

### 5. Verify Tesseract Installation

Verify that Tesseract is properly installed by running:

```bash
tesseract --version
```

You should see the installed version of Tesseract.

## Running the Application 🚀

### Start the Streamlit Application

Run the following command to start the Streamlit app:

```bash
streamlit run app.py
```

### Access the Web Interface

After running the command, Streamlit will start the server, and you'll see a URL in the terminal (usually http://localhost:8501). Open this URL in your web browser to access the Docx-DRS interface.

### Upload Documents

- You can upload up to 3 documents simultaneously (PDF, DOCX, XLSX, PNG, JPG).
- The system will process the documents, and you will be able to search and extract relevant text based on your input.

If you face any issues, feel free to open an issue in the GitHub repository.
