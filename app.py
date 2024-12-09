import streamlit as st
from document_processor import DocumentProcessor
def main():
    st.title("📄 Advanced Document Search System")
    
    # Session state for managing documents
    if 'documents' not in st.session_state:
        st.session_state.documents = []
        st.session_state.file_details = []
        st.session_state.suggestions = []
        st.session_state.search_results = None  # Store search results
        st.session_state.search_query = ""  # Store search query

    # File Upload
    uploaded_files = st.file_uploader(
        "Upload up to 3 Documents (PDF, DOCX, XLSX, PNG, JPG)", 
        type=["pdf", "docx", "xlsx", "png", "jpg"], 
        accept_multiple_files=True
    )
    
    # Process uploaded files
    if uploaded_files:
        # Limit to 3 documents
        uploaded_files = uploaded_files[:3]
        
        # Reset existing documents
        st.session_state.documents.clear()
        st.session_state.file_details.clear()
        st.session_state.suggestions.clear()
        
        # Process each document
        for file in uploaded_files:
            # Extract text
            doc_text = DocumentProcessor.extract_text(file)
            
            # Store document details
            st.session_state.documents.append(doc_text)
            st.session_state.file_details.append({
                'name': file.name,
                'file': file
            })
            
            # Generate suggestions
            suggestions = DocumentProcessor.generate_suggestions(doc_text)
            st.session_state.suggestions.append(suggestions)
    
    # Suggestions and Search Section
    if st.session_state.documents:
        st.header("📋 Document Suggestions")
        
        # Display suggestions for each document
        suggestion_columns = st.columns(len(st.session_state.documents))
        
        for i, (suggestions, doc_detail) in enumerate(zip(st.session_state.suggestions, st.session_state.file_details)):
            with suggestion_columns[i]:
                st.subheader(f"Doc {i+1}: {doc_detail['name']}")
                
                # Clickable suggestions
                for suggestion in suggestions:
                    if st.button(suggestion, key=f"sug_{i}_{suggestion}"):
                        # Update search query from suggestion
                        st.session_state.search_query = suggestion
                        st.session_state.search_scope = "All Documents"
                        search_documents(st.session_state.search_query, st.session_state.search_scope)

        # Custom Search Section
        st.header("🔍 Custom Search")

        # Search input with Enter key support
        search_query = st.text_input(
            "Enter your search query:", 
            value=st.session_state.search_query, 
            on_change=lambda: search_documents(
                st.session_state.search_query, 
                st.session_state.search_scope
            )
        )
        
        # Update query in session state
        st.session_state.search_query = search_query

        # Search scope selection
        search_scope = st.radio(
        "Search Scope", 
        ["All Documents", "Document 1", "Document 2", "Document 3"], 
        key="search_scope")

        # Update session state only if the value changes
        if 'search_scope' not in st.session_state or st.session_state.search_scope != search_scope:
            st.session_state.search_scope = search_scope
                
        # Search button
        if st.button("Search"):
            search_documents(st.session_state.search_query, st.session_state.search_scope)
        
        # Display search results below the search button
        st.header("Search Results")
        if st.session_state.search_results:
            for result in st.session_state.search_results:
                # Passages
                for passage in result['relevant_passages']:
                    st.write(passage)
                
                # Download button
                st.download_button(
                    label="Download Document", 
                    data=st.session_state.file_details[result['document_index']]['file'].getvalue(),
                    file_name=st.session_state.file_details[result['document_index']]['name']
                )
        else:
            st.write("No search results to display.")
    else:
        st.warning("No documents uploaded yet.")
    
    # Sidebar
    with st.sidebar:
        st.title("📚 Document Search Features")
        st.markdown("""
        ## Capabilities:
        - Upload PDF, DOCX, XLSX, PNG, JPG
        - Automatic text extraction
        - Keyword suggestions
        - Flexible document search
        - Direct document download
        """)
        st.write("Made with ❤️ by Harshit")

# Helper function for performing search
def search_documents(query, scope):
    """
    Perform search based on query and scope
    """
    if not query:
        st.warning("Please enter a search query")
        return
    
    # Determine search scope
    if scope == "All Documents":
        search_docs = st.session_state.documents
    else:
        # Map radio selection to document index
        doc_index = int(scope.split()[-1]) - 1
        search_docs = [st.session_state.documents[doc_index]]
    
    # Perform search
    search_results = DocumentProcessor.advanced_search(search_docs, query)
    st.session_state.search_results = search_results    

if __name__ == "__main__":
    main()
