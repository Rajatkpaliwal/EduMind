# 🤖 AI ChatBot Blueprint

An intelligent chatbot built using **Streamlit, LangChain, Groq LLM, Retrieval-Augmented Generation (RAG), and Live Web Search**.

This project was developed for the **NeoStats Challenge — The Chatbot Blueprint: Imagine · Build · Solve**.

The chatbot can answer questions by combining:

* 📄 Knowledge from **uploaded documents (PDF/TXT)**
* 🌐 **Real-time web search**
* 🧠 **LLM reasoning**

---

# 🚀 Features

## 📄 Retrieval-Augmented Generation (RAG)

Users can upload their own documents and ask questions about them.

**How it works:**

1. Upload **PDF or TXT files**
2. Documents are split into smaller chunks
3. Chunks are converted into **vector embeddings**
4. Stored in **FAISS vector database**
5. Relevant chunks are retrieved when a question is asked

**Tech used:**

* HuggingFace Embeddings (`all-MiniLM-L6-v2`)
* FAISS Vector Database
* RecursiveCharacterTextSplitter

---

## 🌐 Live Web Search

When the chatbot cannot find information in uploaded documents, it performs a **live web search**.

**Integration:**

* Google Serper API
* LangChain `GoogleSerperAPIWrapper`

This allows the chatbot to answer **current events and real-time queries**.

---

## ⚡ Response Modes

Users can choose between two response styles:

| Mode         | Description                           |
| ------------ | ------------------------------------- |
| **Concise**  | Short 2–4 sentence answers            |
| **Detailed** | Structured explanations with examples |

This is implemented through **dynamic system prompts**.

---

## 🎨 Modern Chat Interface

The chatbot UI is built using **Streamlit** with custom styling.

Features include:

* Dark themed chat UI
* Chat history
* File upload for RAG
* RAG status indicator
* Clear chat button

---

# 🏗 Architecture

```
User Query
     ↓
Streamlit Chat Interface
     ↓
get_chat_response()

Retrieves Context From:
 ├── RAG Retriever (FAISS + HuggingFace)
 └── Web Search (Serper API)

     ↓
Prompt Augmentation
(Document Context + Web Results)

     ↓
Groq LLM (llama-3.1-8b)

     ↓
Final AI Response
```

---

# 📂 Project Structure

```
project/
│
├── config/
│   └── config.py          # API keys & configuration
│
├── models/
│   ├── llm.py             # Groq LLM initialization
│   └── embeddings.py      # HuggingFace embedding model
│
├── utils/
│   ├── rag.py             # Document loading + retrieval
│   ├── web_search.py      # Google Serper web search
│   └── agent.py           # LangChain agent tools
│
├── app.py                 # Main Streamlit application
├── requirements.txt
├── .env.example
└── README.md
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Rajatkpaliwal/EduMind.git
cd EduMind
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key
SERPER_API_KEY=your_serper_api_key
```

You can obtain API keys from:

* Groq → https://console.groq.com/keys
* Serper → https://serper.dev

⚠️ Never commit `.env` files to GitHub.

---

# ▶️ Run the Application

Start the Streamlit server:

```bash
streamlit run app.py
```

The app will open in your browser.

---

# 📄 How to Use

### Ask General Questions

Type a question directly into the chat interface.

Example:

```
Explain transformers in deep learning
```

---

### Ask Questions About Documents

1. Upload **PDF or TXT files** from the sidebar
2. Click **Process Documents**
3. Ask questions related to the uploaded content

Example:

```
Summarize chapter 2 of the uploaded notes
```

---

### Ask Real-Time Questions

Example:

```
Latest news about OpenAI
```

The chatbot will automatically perform a **web search**.

---

# 🧠 Key Concepts Used

### Retrieval-Augmented Generation (RAG)

Instead of relying only on the LLM’s training data, the chatbot retrieves relevant context from external documents.

---

### Prompt Augmentation

The system augments the prompt with:

* Document context
* Web search results

before sending the request to the LLM.

---

### Vector Search

Semantic search is performed using **FAISS vector similarity search**.

---

# 🛠 Technologies Used

| Category   | Technology       |
| ---------- | ---------------- |
| LLM        | Groq (Llama-3.1) |
| Framework  | LangChain        |
| Embeddings | HuggingFace      |
| Vector DB  | FAISS            |
| Web Search | Google Serper    |
| UI         | Streamlit        |
| Language   | Python           |

---

# 👨‍💻 Author

Rajat Kumar Paliwal

Computer Science & Engineering
