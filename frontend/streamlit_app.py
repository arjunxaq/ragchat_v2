import streamlit as st
import requests
import time
import os

# Backend API URL - get from environment or use default
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

st.set_page_config(page_title="RAGChat", page_icon="💬", layout="wide")

st.title("💬 RAGChat - Chat with your PDF")
st.markdown("Upload a PDF and ask questions about its content!")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False
if "collection_id" not in st.session_state:
    st.session_state.collection_id = None

# Sidebar for PDF upload
with st.sidebar:
    st.header("📄 Upload PDF")
    
    # Show backend status
    try:
        health_response = requests.get(f"{BACKEND_URL}/health", timeout=2)
        if health_response.status_code == 200:
            st.success("🟢 Backend Connected")
        else:
            st.warning("🟡 Backend Responding (Check)")
    except:
        st.error("🔴 Backend Unavailable")
        st.info(f"Trying to connect to: {BACKEND_URL}")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Process PDF", type="primary"):
            with st.spinner("Processing PDF..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=120)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.collection_id = result["collection_id"]
                        st.session_state.pdf_uploaded = True
                        st.session_state.messages = []
                        st.success(f"✅ PDF processed! {result['chunks']} chunks created.")
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {str(e)}")
                    st.info("Make sure both frontend and backend containers are running in the same Docker network.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    if st.session_state.pdf_uploaded:
        st.success("✅ PDF ready for questions!")
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        if st.button("Reset (Upload new PDF)"):
            st.session_state.messages = []
            st.session_state.pdf_uploaded = False
            st.session_state.collection_id = None
            st.rerun()

# Main chat interface
if not st.session_state.pdf_uploaded:
    st.info("👈 Please upload a PDF from the sidebar to start chatting!")
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:**")
                        st.text(source[:300] + "..." if len(source) > 300 else source)
                        st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your PDF..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/chat",
                        json={
                            "question": prompt,
                            "collection_id": st.session_state.collection_id
                        },
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        answer = result["answer"]
                        sources = result.get("sources", [])
                        
                        st.markdown(answer)
                        
                        if sources:
                            with st.expander("📚 View Sources"):
                                for i, source in enumerate(sources, 1):
                                    st.markdown(f"**Source {i}:**")
                                    st.text(source[:300] + "..." if len(source) > 300 else source)
                                    st.divider()
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                    else:
                        error_msg = f"Error: {response.json().get('detail', 'Unknown error')}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })
                except Exception as e:
                    error_msg = f"Connection error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })