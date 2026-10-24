
import streamlit as st
from pypdf import PdfReader
from google import genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI College Knowledge Assistant",
    page_icon="📚"
)

st.title("📚 AI College Knowledge Assistant")
st.write("Upload your notes and ask questions using AI.")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_file = st.file_uploader(
    "Upload your PDF notes",
    type=["pdf"]
)

if uploaded_file:
    reader = PdfReader(uploaded_file)
    chunks = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        chunk_size = 500
        overlap = 100
        start = 0

        while start < len(text):
            chunk_text = text[start:start + chunk_size]

            if chunk_text.strip():
                chunks.append({
                    "page": page_num,
                    "text": chunk_text
                })

            start += chunk_size - overlap

    if chunks:
        st.success(f"Created {len(chunks)} text chunks.")

        with st.spinner("Loading semantic search model..."):
            model = load_model()

        texts = [chunk["text"] for chunk in chunks]

        with st.spinner("Creating embeddings..."):
            embeddings = model.encode(texts)

        # Display previous messages
        st.subheader("💬 Chat")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input("Ask a question about your notes")

        if question:
            st.session_state.messages.append({
                "role": "user",
                "content": question
            })

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Finding relevant notes..."):
                    query_embedding = model.encode([question])

                    scores = cosine_similarity(
                        query_embedding, embeddings
                    )[0]

                    top_indices = scores.argsort()[::-1][:3]

                    context = "\n\n".join(
                        f"Page {chunks[idx]['page']}:\n"
                        f"{chunks[idx]['text']}"
                        for idx in top_indices
                    )

                prompt = f"""
Answer using only the context from the uploaded notes.
If the answer is not available in the context, say that
the notes do not contain enough information.

Context:
{context}

Question: {question}
"""

                try:
                    client = genai.Client(
                        api_key=st.secrets["GEMINI_API_KEY"]
                    )

                    response = client.models.generate_content(
                        model="gemini-3.8-flash",
                        contents=prompt
                    )

                    answer = response.text or (
                        "No text answer was returned."
                    )

                    st.markdown(answer)

                    st.markdown("**📚 Sources**")
                    source_pages = sorted({
                        chunks[idx]["page"]
                        for idx in top_indices
                    })
                    st.write(", ".join(
                        f"Page {page}" for page in source_pages
                    ))

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:
                    st.error(f"{type(e).__name__}: {e}")

        with st.expander("🔎 View relevant passages"):
            for idx in top_indices if question else []:
                chunk = chunks[idx]
                st.markdown(f"**Page:** {chunk['page']}")
                st.markdown(
                    f"**Similarity:** {scores[idx]:.2f}"
                )
                st.write(chunk["text"])
                st.divider()

        if st.button("🗑️ Clear chat"):
            st.session_state.messages = []
            st.rerun()

    else:
        st.warning("No readable text found in this PDF.")
else:
    st.info("Upload a PDF to start asking questions.")


