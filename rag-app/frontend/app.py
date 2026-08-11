import os
import streamlit as st
import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.title("Simple RAG")

# Document Upload Section
st.subheader("Upload Document")
uploaded_file = st.file_uploader("Choose File (PDF or TXT)", type=["pdf", "txt", "md"])

if st.button("Upload"):
    if uploaded_file:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        res = requests.post(f"{BACKEND_URL}/documents/upload", files=files)
        if res.ok:
            st.success(res.json()["message"])
        else:
            st.error(res.json().get("detail", "Upload failed."))
    else:
        st.warning("Please choose a file first.")

st.divider()

# Question Answering Section
st.subheader("Ask a Question")
question = st.text_input("Enter your question:")

if st.button("Ask"):
    if question.strip():
        res = requests.post(f"{BACKEND_URL}/query", json={"question": question.strip()})
        if res.ok:
            data = res.json()
            st.subheader("Answer")
            st.info(data["answer"])

            if data.get("context"):
                st.subheader("Sources")
                for idx, chunk in enumerate(data["context"], 1):
                    with st.expander(f"Source Chunk #{idx}"):
                        st.write(chunk)
        else:
            st.error(res.json().get("detail", "Query failed."))
    else:
        st.warning("Please enter a question.")
