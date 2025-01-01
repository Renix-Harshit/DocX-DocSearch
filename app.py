import streamlit as st
import os
from document_processor import DocumentProcessor

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
        st.session_state.custom_search_active = False

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

            # V1-vector store
            st.session_state.vector_store, st.session_state.chunks, st.session_state.chunk_map = (
                st.session_state.doc_processor.create_vector_store(st.session_state.documents)
            )
    if st.session_state.documents:
        st.header("📋 Document Suggestions")
        suggestion_options = ["Select a suggestion..."] + st.session_state.all_suggestions

        def suggestion_selected():
            if st.session_state.selected_suggestion != "Select a suggestion...":
                st.session_state.search_query = st.session_state.selected_suggestion
                st.session_state.search_results = st.session_state.doc_processor.semantic_search(
                    st.session_state.search_query,
                    st.session_state.vector_store,
                    st.session_state.chunks,
                    st.session_state.chunk_map
                )

        selected_suggestion = st.selectbox(
            "Quick Search Suggestions",
            options=suggestion_options,
            key="selected_suggestion",
            on_change=suggestion_selected
        )

        if st.button("Custom Search"):
            st.session_state.custom_search_active = True

        if st.session_state.custom_search_active:
            custom_query = st.text_input("Enter your custom query:", key="custom_search_input")

            if st.button("Search"):
                st.session_state.search_query = custom_query
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
                distance = result['distance']
                st.write(f"**Document: {doc_name}**")
                st.write(f"Similarity Score: {distance:.4f}")
                for qa in result['questions_answers']:
                    st.write(f"**Question:** {qa['question']}")
                    st.write(f"**Answer:** {qa['answer']}")

                st.download_button(
                    label=f"Download {doc_name}",
                    data=st.session_state.file_details[result['document_index']]['file'].getvalue(),
                    file_name=doc_name,
                    key=f"download_{doc_name}" #unique key for each download button
                )

                st.markdown("---") # Separator between search results
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

if __name__ =="__main__":
    main()
