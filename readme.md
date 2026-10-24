
# AI College Knowledge Assistant 📚

An AI-powered college study assistant that answers questions from uploaded PDF notes.

## Features
- Upload PDF notes
- Ask questions using Gemini AI
- Semantic search with Sentence Transformers
- View relevant passages and source pages
- Chat interface built with Streamlit

## Technologies
- Python
- Streamlit
- Gemini API
- PyPDF
- Sentence Transformers
- Scikit-learn

## Run Locally

Install dependencies:

pip install -r requirements.txt

Create `.streamlit/secrets.toml` and add:

GEMINI_API_KEY = "your-api-key"

Run the app:

streamlit run main.py
