# 💬 InkLink

**Turn every page into a conversation.**

InkLink is a powerful RAG-powered Streamlit application that allows you to upload PDF documents and have intelligent conversations about their content using Anthropic's Claude AI. Whether you're researching, studying, or analyzing documents, InkLink makes it easy to extract insights and get answers from your PDFs through advanced vector search and semantic retrieval.

## ✨ Features

### 🚀 Advanced RAG (Retrieval-Augmented Generation)
- 🧠 **Vector Database**: ChromaDB for efficient document storage and retrieval
- 🔍 **Semantic Search**: Find relevant content using sentence transformers
- 📝 **Smart Chunking**: Intelligent text splitting with overlap for context preservation
- 🎯 **Contextual Responses**: Only uses relevant document sections for each query

### 💻 Core Functionality
- 📁 **Multiple PDF Upload**: Upload single or multiple PDF files simultaneously
- 🤖 **Claude AI Integration**: Powered by Anthropic's Claude 3.5 Sonnet for intelligent responses
- 💬 **Interactive Chat**: Natural conversation interface with persistent chat history
- 📊 **Document Context**: Maintains context across all uploaded documents
- 🎨 **Clean UI**: Intuitive and user-friendly Streamlit interface
- 🔄 **Session Management**: Chat history persists during your session
- 🧹 **Easy Reset**: Clear chat history or upload new documents anytime
- 📈 **Database Stats**: Track processed documents and chunks

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Anthropic API key (get one from [Anthropic Console](https://console.anthropic.com/))

### Installation

1. **Clone or download the project**:
   ```bash
   git clone <your-repo-url>
   cd chatwithpdf
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

## 📋 Usage

1. **Upload PDFs**: Use the sidebar to upload one or multiple PDF files
2. **Process Documents**: Click "Process PDFs" to extract text from your documents
3. **Start Chatting**: Ask questions about your documents in the chat interface
4. **Get Insights**: Claude will provide intelligent responses based on your document content

### Example Questions

- "What are the main points discussed in this document?"
- "Can you summarize the key findings?"
- "What does the document say about [specific topic]?"
- "Compare the arguments presented in different sections"

## 👩‍💻 Connect with the Creator

<div align="center">

**Made with ❤️ by Geetanshi Goel**

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/geetanshi0205)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/geetanshi-goel-49ba5832b/)

</div>

---

## 🛠️ Technical Details

### Dependencies

#### Core Dependencies
- **Streamlit**: Modern web app framework for Python
- **PyPDF2**: PDF text extraction library
- **Anthropic**: Official Python client for Claude API
- **python-dotenv**: Environment variable management

#### RAG Technology Stack
- **ChromaDB**: Vector database for document storage and retrieval
- **Sentence Transformers**: Generate embeddings for semantic search
- **LangChain**: Text splitting and chunking utilities
- **all-MiniLM-L6-v2**: Lightweight embedding model for fast processing

### Architecture

The application uses advanced RAG (Retrieval-Augmented Generation) architecture:

1. **PDF Processing**: Extracts text from uploaded PDF files using PyPDF2
2. **Document Chunking**: Splits documents into overlapping chunks for better context preservation
3. **Vector Embeddings**: Generates semantic embeddings using sentence transformers
4. **Vector Storage**: Stores embeddings in ChromaDB for fast similarity search
5. **Semantic Retrieval**: Searches for relevant chunks based on user queries
6. **Context Assembly**: Combines relevant chunks from multiple documents
7. **AI Generation**: Claude generates responses using only relevant document context
8. **Chat Interface**: Displays conversation with real-time responses and database stats

#### RAG Workflow
```
User Query → Embedding → Similarity Search → Relevant Chunks → Claude AI → Response
```

### File Structure

```
chatwithpdf/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env.example       # Environment variable template
├── .env               # Your environment variables (create this)
└── README.md          # This file
```

## 🎯 Features in Detail

### 🧠 Advanced RAG Implementation
**Smart Document Processing:**
- Documents are split into 1000-character chunks with 200-character overlap
- Each chunk is converted to semantic embeddings using all-MiniLM-L6-v2
- Embeddings stored in ChromaDB with metadata (filename, chunk index)

**Intelligent Retrieval:**
- User queries are embedded and compared against document chunks
- Top 5 most relevant chunks are retrieved using cosine similarity
- Only relevant content is sent to Claude AI, improving accuracy and reducing costs

**Context-Aware Responses:**
- Claude receives only the most pertinent information
- Reduces hallucinations by grounding responses in actual document content
- Maintains conversation history for natural dialogue flow

### 📊 Multi-PDF Support
- Upload and process multiple PDF files simultaneously
- Each document is tracked separately in the vector database
- Cross-document queries supported for comprehensive analysis
- Document source attribution in responses

### 🎛️ Database Management
- Real-time stats showing processed documents and chunks
- Ability to clear individual chat history or entire document database
- Persistent storage during session for fast retrieval
- Duplicate document detection to avoid reprocessing

### 🎨 Enhanced User Interface
- Clean, modern Streamlit interface with RAG status indicators
- Sidebar shows database statistics and management options
- Real-time processing feedback with chunk count information
- Visual indicators for RAG-powered responses

## 🔧 Configuration

### Environment Variables

Create a `.env` file with:

```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### Model Configuration

The application uses Claude 3.5 Sonnet by default. You can modify the model in `app.py`:

```python
model="claude-3-5-sonnet-20241022"
```

## 🚨 Troubleshooting

### Common Issues

1. **API Key Error**: Make sure your `.env` file contains a valid Anthropic API key
2. **PDF Processing Error**: Ensure your PDF files are text-based (not scanned images)
3. **Import Error**: Run `pip install -r requirements.txt` to install all dependencies

### Error Messages

- `"Please set your ANTHROPIC_API_KEY environment variable"`: Add your API key to `.env`
- `"model: claude-3-sonnet-20240229"`: Model name issue (should be fixed in current version)

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues and pull requests.

## 📜 License

This project is open source and available under the MIT License.

## 👩‍💻 About

**InkLink** was created by **Geetanshi Goel** to make document analysis and research more accessible through AI-powered conversations and advanced RAG technology.

---

*Built with ❤️ using Streamlit, Claude AI, and ChromaDB*