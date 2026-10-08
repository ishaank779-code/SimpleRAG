"""
SimpleRAG Frontend - Streamlit Chat UI
Run: streamlit run frontend.py
"""

import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="SimpleRAG Chat",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 SimpleRAG – Document Q&A")
st.caption("Upload documents → Ask questions → Get grounded answers")

# Sidebar
with st.sidebar:
    st.header("📄 Documents")
    
    # Health check
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        st.success(f"API: {health.get('status', 'ok')}")
        st.caption(f"FAISS: {health.get('faiss', '-')}")
        st.caption(f"MySQL: {health.get('mysql', '-')}")
    except Exception as e:
        st.error(f"API not reachable: {e}")
        st.stop()

    st.divider()

    # Upload
    uploaded = st.file_uploader("Upload PDF / TXT / MD", type=["pdf", "txt", "md"])
    if uploaded and st.button("Upload & Index", type="primary"):
        with st.spinner("Uploading & embedding..."):
            files = {"file": (uploaded.name, uploaded.getvalue())}
            res = requests.post(f"{API_URL}/documents/upload", files=files, timeout=120)
            if res.status_code == 200:
                st.success(f"Uploaded: {res.json().get('original_name')} ({res.json().get('num_chunks')} chunks)")
                st.rerun()
            else:
                st.error(res.text)

    st.divider()

    # List documents
    docs = []
    try:
        docs = requests.get(f"{API_URL}/documents", timeout=10).json()
        if docs:
            st.subheader("Indexed docs")
            for d in docs:
                st.write(f"**{d['original_name']}** — {d['num_chunks']} chunks ({d['status']})")
        else:
            st.info("No documents yet. Upload one!")
    except Exception:
        pass

    st.divider()
    doc_filter = st.selectbox(
        "Search in",
        options=[None] + [d["id"] for d in docs] if docs else [None],
        format_func=lambda x: "All documents" if x is None else next(
            (d["original_name"] for d in docs if d["id"] == x), str(x)
        ),
    )

# Chat history in session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"**{s.get('document_name', '')}** (score: {s.get('score', 0)})")
                    st.caption(s.get("content", "")[:300])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            payload = {"question": prompt, "document_id": doc_filter}
            res = requests.post(f"{API_URL}/query", json=payload, timeout=60)
            if res.status_code == 200:
                data = res.json()
                answer = data.get("answer", "No answer")
                sources = data.get("sources", [])
                st.markdown(answer)
                if sources:
                    with st.expander("Sources"):
                        for s in sources:
                            st.markdown(f"**{s.get('document_name', '')}** (score: {s.get('score', 0)})")
                            st.caption(s.get("content", "")[:300])
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            else:
                st.error(f"Error: {res.text}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"Error: {res.text}"}
                )