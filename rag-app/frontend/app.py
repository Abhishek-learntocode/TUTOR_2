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
        if msg.get("sources"):
            with st.expander("Sources"):
                for idx, chunk in enumerate(msg["sources"], 1):
                    st.markdown(f"**Chunk #{idx}:**\n{chunk}")

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
                    sources = data.get("context", [])

                    st.markdown(answer)
                    if sources:
                        with st.expander("Sources"):
                            for idx, chunk in enumerate(sources, 1):
                                st.markdown(f"**Chunk #{idx}:**\n{chunk}")

                    st.session_state.messages.append({"role": "user", "content": prompt})
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })
                else:
                    st.error(res.json().get("detail", "Query failed."))
            except Exception as e:
                st.error(f"Connection error: {e}")

