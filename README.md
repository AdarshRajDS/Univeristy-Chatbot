
# 🎓 University Chatbot — AI-Powered RAG System

**Live Demo:** [https://univeristy-chatbot-eippytsakrfvuyac8qi8ch.streamlit.app/](https://univeristy-chatbot-eippytsakrfvuyac8qi8ch.streamlit.app/)

An AI chatbot that answers university-related questions using **LLMs**, **Retrieval-Augmented Generation (RAG)**, and **automated web crawling**.
This system ingests university documentation, builds semantic embeddings, and retrieves context-rich responses in real time.

---

## 🚀 Features

* 🔍 **Web Crawling & Indexing** (Tavily crawler, depth-2)
* 🧠 **RAG Pipeline** using LangChain
* ⚡ **Groq-powered LLM inference**
* 📚 **Embeddings & Vector Search** using HuggingFace + ChromaDB
* 🌐 **Interactive Streamlit Interface**
* ⚙️ **Fully containerized + lockfile-based environment using uv**
* 📎 Future support for **custom PDF/document uploads**
* ☁️ Deployed on **Streamlit Cloud**

---

## 🛠 Tech Stack

### **Languages & Core**

* Python 3.12

### **Frontend / UI**

* Streamlit

### **AI / LLM / RAG**

* LangChain
* Groq LLM
* LangSmith (tracing)
* Sentence-Transformers
* HuggingFace Embeddings
* ChromaDB
* Pinecone (optional integration)

### **Crawling & Ingestion**

* Tavily (Crawl + Extract)
* Recursive text splitting
* Certifi, Requests, SSL utilities

### **DevOps**

* uv (dependency and lockfile-based environment)
* python-dotenv
* GitHub for version control
* Streamlit Cloud deployment

---

## 📁 Project Structure

```
📦 university-chatbot
├── backend/
│   ├── core.py               # Main RAG pipeline (retriever + LLM + embeddings)
│   ├── consts.py             # Constants (index name, configs)
│   └── __init__.py
│
├── ingestion.py              # Web crawler + document ingestion + embedding pipeline
├── chroma_db/                # Vector database (local Chroma instance)
├── main.py                   # Streamlit UI
│
├── static/                   # Icons, images, assets
├── pyproject.toml            # Project and dependency definitions
├── uv.lock                   # Reproducible lockfile
├── requirements.txt          # Exported requirements (for compatibility)
│
├── README.md
└── LICENSE
```

---

## ⚙️ How It Works

### **1️⃣ Web Crawling**

The system uses **Tavily** to crawl university documentation pages (depth: 2).
Extracted pages → cleaned → stored as documents.

### **2️⃣ Text Chunking**

Content is split using LangChain’s `RecursiveCharacterTextSplitter` for optimal chunk sizes.

### **3️⃣ Embedding Generation**

Embeddings generated with:

* `sentence-transformers`
* HuggingFace models
* Optionally OpenAI/Groq embedding models

Stored in **ChromaDB** or **Pinecone**.

### **4️⃣ RAG Query Pipeline**

User question → Retrieve top-k documents → Context-packed prompt → Groq LLM → Final answer.

### **5️⃣ Streamlit Interface**

Clean UI that supports:

* A sidebar for configuration
* Query box
* Answer formatting
* Document sources preview (optional)

---

## 🧩 Running Locally

### **1. Clone the repository**

```bash
git clone https://github.com/AdarshRajDS/Univeristy-Chatbot.git
cd Univeristy-Chatbot
```

### **2. Install uv (if you don’t have it)**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### **3. Sync environment**

```bash
uv sync
```

### **4. Add your environment variables**

Create a `.env` file:

```
GROQ_API_KEY=your_key
TAVILY_API_KEY=your_key
OPENAI_API_KEY=optional_key
LANGCHAIN_API_KEY=your_key
```

### **5. Run the app**

```bash
streamlit run main.py
```

---

## 🧭 Roadmap

* 📄 Support **PDF / Document upload**
* ⬆ Increase crawling depth beyond level 2
* 📊 Add analytics dashboard
* 🤖 Multi-university support
* 🔍 Evaluate other embedding models (E5-large, BAAI, Cohere)
* 🚀 Add response citations and source preview

---

## 🤝 Contributing

Pull requests are welcome!
Feel free to open an issue for suggestions or improvements.

---

## 📝 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

Thanks to the LangChain, Groq, HuggingFace, and Tavily teams for their incredible open-source work.

