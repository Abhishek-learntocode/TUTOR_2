import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Simple RAG Chat", page_icon="💬", layout="centered")

st.title("Simple RAG")
st.caption("Minimal ChatGPT-style interface powered by FastAPI & LangGraph")

# Initialize conversation history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Document Upload Control
with st.sidebar:
    st.subheader("📎 Attach Files")
    uploaded_files = st.file_uploader(
        "Select PDF or Text files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        key="file_uploader",
    )

    if uploaded_files:
        st.markdown("**Selected Files:**")
        for f in uploaded_files:
            st.markdown(f"✓ `{f.name}`")

        selected_type = st.selectbox(
            "Document Type:",
            ["BOOK", "EXAM_PAPER"],
            key="doc_type_select",
        )

        if st.button("Upload Files", use_container_width=True, type="primary"):
            success_count = 0
            with st.spinner("Processing & indexing files..."):
                for file in uploaded_files:
                    try:
                        files = {"file": (file.name, file.getvalue(), file.type)}
                        data = {"doc_type": selected_type}
                        res = requests.post(f"{BACKEND_URL}/documents/upload", files=files, data=data)
                        if res.ok:
                            success_count += 1
                        else:
                            st.error(f"Failed to upload '{file.name}': {res.json().get('detail')}")
                    except Exception as e:
                        st.error(f"Error uploading '{file.name}': {e}")

            if success_count > 0:
                msg = f"Successfully uploaded {success_count} file(s) as [{selected_type}]!"
                st.toast(msg, icon="✅")
                st.success(msg)

    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Display chat message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("📚 Source Citations"):
                for idx, src in enumerate(msg["citations"], 1):
                    doc_id = src.get("source_filename") or src.get("document_id", "Doc")
                    page_info = f", Page {src['page_number']}" if src.get("page_number") is not None else ""
                    sec_info = f" ({src['section']})" if src.get("section") else ""
                    st.markdown(f"**[{idx}] `{doc_id}`**{page_info}{sec_info} `chunk:{src.get('chunk_id')}`")
        elif msg.get("sources"):
            with st.expander("Sources"):
                for idx, chunk in enumerate(msg["sources"], 1):
                    st.markdown(f"**Chunk #{idx}:**\n{chunk}")
        if msg.get("metrics"):
            with st.expander("⚡ Latency Metrics"):
                for k, v in msg["metrics"].items():
                    st.caption(f"**{k}**: {v:.4f}s")

# Chat input prompt
if prompt := st.chat_input("Ask anything..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching & generating answer..."):
            try:
                history_payload = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]
                res = requests.post(
                    f"{BACKEND_URL}/query",
                    json={"question": prompt, "chat_history": history_payload},
                )
                if res.ok:
                    data = res.json()
                    answer = data.get("answer", "")
                    raw_context = data.get("context", [])
                    citations = data.get("sources", [])
                    metrics = data.get("metrics", {})

                    st.markdown(answer)

                    if citations:
                        with st.expander("📚 Source Citations"):
                            for idx, src in enumerate(citations, 1):
                                doc_id = src.get("source_filename") or src.get("document_id", "Doc")
                                page_info = f", Page {src['page_number']}" if src.get("page_number") is not None else ""
                                sec_info = f" ({src['section']})" if src.get("section") else ""
                                st.markdown(f"**[{idx}] `{doc_id}`**{page_info}{sec_info} `chunk:{src.get('chunk_id')}`")
                    elif raw_context:
                        with st.expander("Sources"):
                            for idx, chunk in enumerate(raw_context, 1):
                                st.markdown(f"**Chunk #{idx}:**\n{chunk}")

                    if metrics:
                        with st.expander("⚡ Latency Metrics"):
                            for k, v in metrics.items():
                                st.caption(f"**{k}**: {v:.4f}s")

                    st.session_state.messages.append({"role": "user", "content": prompt})
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                        "sources": raw_context,
                        "metrics": metrics,
                    })
                else:
                    st.error(res.json().get("detail", "Query failed."))
            except Exception as e:
                st.error(f"Connection error: {e}")

