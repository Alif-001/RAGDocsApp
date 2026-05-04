# 📚 PDF Question Answering System using RAG

[![Live Demo](https://img.shields.io/badge/Live_Demo-Try_It_Now-blue?style=for-the-badge&logo=streamlit)](https://ragdocsapp.streamlit.app/)
<br>
**🔗 Live Link:** [https://ragdocsapp.streamlit.app/](https://ragdocsapp.streamlit.app/)

> **A robust Retrieval-Augmented Generation (RAG) application that allows users to seamlessly converse with their PDF documents.**

## 🎯 Executive Summary
This project is an advanced, conversational AI tool designed to read, understand, and answer questions based on the content of uploaded PDF documents. By leveraging **Retrieval-Augmented Generation (RAG)**, it limits Large Language Model (LLM) "hallucinations" by strictly confining answers to the exact context retrieved from the user's secure documents.

## 🚀 Key Highlights & Skills Demonstrated
- **Applied Generative AI & Prompt Engineering**: Interfaced with fast Groq/Google LLM endpoints to synthesize answers dynamically while forcing strict constraints to use context-only facts.
- **RAG Architecture Implementation**: Successfully bridged the gap between static documents and LLMs by indexing, querying, and retrieving semantic text segments.
- **Vector Search & Embeddings**: Utilized Hugging Face's `sentence-transformers` (`all-MiniLM-L6-v2`) locally to accurately encode text chunks, and implemented a custom cosine similarity search engine entirely mathematically via `NumPy`.
- **Full-Stack Prototyping**: Developed an interactive, stateful chat frontend entirely in Python using **Streamlit**, featuring customizable parameters (chunk sizes, overlap, Top-k search). 

## 🧠 System Architecture
1. **Document Ingestion**: Extracts raw text from complex PDF files utilizing `PyPDF2`.
2. **Text Processing**: Segments long documents into continuous, overlapping chunks to preserve contextual flow without exceeding LLM token constraints.
3. **Embedding Generation**: Safely encodes text segments into normalized multi-dimensional vectors using local open-source Machine Learning models.
4. **Semantic Search Engine**: Computes vector similarity in real-time to find and retrieve the most relevant context blocks for any given user query.
5. **LLM Synthesis**: Injects the retrieved chunks transparently as augmented context to an LLM, yielding highly accurate and deterministic responses with sources cited.

## 🛠️ Technology Stack
- **Language**: Python
- **AI/ML Methods**: Retrieval-Augmented Generation (RAG), Semantic Vector Search, Document Chunking
- **Models**: `sentence-transformers/all-MiniLM-L6-v2`, Groq API (Llama3/Mixtral) / Google Gemini
- **Libraries**: Streamlit (UI), PyPDF2 (Parsing), NumPy (Vector Ops), Requests

## 🚦 Running It Locally

1. **Clone the repository**:
   ```bash
   git clone <your-repository-url>
   cd RAG-based_PDF_Question_Answering_System
   ```
2. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   # OR
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
4. **Launch the application**:
   ```bash
   streamlit run app.py
   ```

## 📸 Demo & Use Case
Users can upload any technical manual, research paper, or legal document, and interact with the text. The UI distinctly shows what segments of text the LLM used to derive its answers, keeping transparency and trust at the forefront.

---
*This project was built to showcase practical knowledge spanning modern AI integration, backend vector logic, and frontend rapid prototyping.*
