# AnantaAI - IISc M.Mgt QA System

An intelligent Question-Answering system for the IISc Master of Management program, powered by advanced AI models and semantic search.

## 🚀 Features

- **Advanced QA Engine**: Powered by Qwen2.5 model with semantic search capabilities
- **Multi-Interface Support**: Both Streamlit web app and React frontend
- **Web Scraping**: Automatic content extraction from IISc websites
- **Semantic Search**: FAISS-based vector search with sentence transformers
- **RESTful API**: FastAPI backend for integration
- **Real-time Responses**: Fast, accurate answers about admissions, placements, curriculum

## 🏗️ Architecture

- **Backend**: FastAPI server with QA processing
- **Frontend**: React TypeScript app with Tailwind CSS
- **Core Engine**: Advanced QA system with semantic search
- **Data Sources**: FAQ data + web scraping from official IISc sites
- **Models**: Sentence transformers + Qwen2.5 for generation

## 📋 Prerequisites

- Python 3.8+
- Node.js 16+ (for frontend)
- CUDA-compatible GPU (optional, for faster processing)

## 🛠️ Installation

### Backend Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd AnantaAI
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

### Frontend Setup

1. **Navigate to frontend directory**
```bash
cd frontend
```

2. **Install Node.js dependencies**
```bash
npm install
```

## 🚀 Usage

### Option 1: Streamlit Interface (Recommended)

```bash
streamlit run app.py
```

Access the application at `http://localhost:8501`

### Option 2: Full Stack (React + FastAPI)

1. **Start the backend server**
```bash
python -m backend.main
```

2. **Start the frontend development server**
```bash
cd frontend
npm run dev
```

Access the application at `http://localhost:3000`

### Option 3: Command Line Interface

```bash
python qna.py "What are the eligibility criteria for M.Mgt?"
```

## 📊 Data Sources

The system uses multiple data sources:

- **context.json**: Curated FAQ data
- **sources.json**: Web sources configuration
- **Web Scraping**: Real-time content from IISc websites
- **faq_data.py**: Structured FAQ categories

## 🔧 Configuration

Key configuration options in the code:

- **Model Settings**: Temperature, max tokens, context window
- **Search Parameters**: Top-k results, confidence thresholds
- **Data Sources**: URLs and categories in `sources.json`

## 📁 Project Structure

```
AnantaAI/
├── app.py                 # Streamlit web interface
├── qna.py                 # Core QA engine
├── webscrapper.py         # Web scraping utilities
├── faq_data.py           # FAQ data structure
├── context.json          # Curated FAQ content
├── sources.json          # Web sources configuration
├── requirements.txt      # Python dependencies
├── backend/
│   └── main.py          # FastAPI server
└── frontend/
    ├── src/
    │   ├── components/  # React components
    │   ├── hooks/       # Custom hooks
    │   └── App.tsx      # Main app component
    ├── package.json     # Node.js dependencies
    └── vite.config.ts   # Vite configuration
```

## 🤖 AI Models Used

- **Sentence Transformers**: `all-MiniLM-L6-v2` for semantic embeddings
- **Qwen2.5**: Advanced language model for answer generation
- **FAISS**: Facebook AI Similarity Search for vector indexing

## 🔍 How It Works

1. **Data Ingestion**: Loads FAQ data and scrapes web content
2. **Semantic Indexing**: Creates vector embeddings for all content
3. **Query Processing**: Converts user questions to embeddings
4. **Similarity Search**: Finds most relevant content using FAISS
5. **Answer Generation**: Uses Qwen2.5 to generate contextual answers
6. **Response Formatting**: Returns structured answers with sources

## 📈 Performance

- **Response Time**: < 2 seconds for most queries
- **Accuracy**: High relevance through semantic search
- **Coverage**: Comprehensive IISc M.Mgt program information
- **Scalability**: Efficient vector search with FAISS

## 🛡️ Error Handling

- Graceful fallbacks for model loading failures
- Robust web scraping with retry mechanisms
- Input validation and sanitization
- Comprehensive logging for debugging

## 🔧 Development

### Adding New Data Sources

1. Update `sources.json` with new URLs and categories
2. The system will automatically scrape and index new content
3. Test with relevant queries

### Customizing the Model

1. Modify `QAConfig` in `qna.py`
2. Adjust temperature, max_tokens, and other parameters
3. Restart the application

### Frontend Development

```bash
cd frontend
npm run dev    # Development server
npm run build  # Production build
npm run preview # Preview production build
```

## 📝 API Documentation

### FastAPI Endpoints

- `POST /api/query`: Submit a question
- `GET /api/categories`: Get available categories

### Request Format

```json
{
  "text": "What are the eligibility criteria?",
  "max_results": 3
}
```

### Response Format

```json
{
  "answer": "Generated answer text",
  "confidence": 0.95,
  "sources": [...],
  "processing_time": 1.23
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- IISc Management Studies Department
- Hugging Face for model hosting
- Facebook AI for FAISS
- The open-source community

## 📞 Support

For questions or issues:
- Create an issue on GitHub
- Check the documentation
- Review the FAQ section

---

**Note**: The web scraper works excellently but may rely heavily on the OCCAP website due to its comprehensive information content.
