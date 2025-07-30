# 🚀 Quick Start Guide - AnantaAI

Get up and running with AnantaAI in under 5 minutes!

## ⚡ Super Quick Start

### Option 1: One-Command Start (Recommended)

**Linux/Mac:**
```bash
./start.sh
```

**Windows:**
```cmd
start.bat
```

### Option 2: Manual Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the app:**
```bash
streamlit run app.py
```

3. **Open your browser:**
Navigate to `http://localhost:8501`

## 🧪 Test the System

Run the test suite to ensure everything works:

```bash
python test_system.py
```

## 🎯 Try These Sample Questions

Once the app is running, try asking:

- "What are the eligibility criteria for M.Mgt?"
- "Tell me about placement statistics"
- "How long is the program duration?"
- "What is the fee structure?"
- "What companies visit for placements?"

## 🔧 Alternative Interfaces

### Command Line Interface
```bash
python qna.py "What are the admission requirements?"
```

### FastAPI Backend + React Frontend
```bash
# Terminal 1: Start backend
cd backend && python main.py

# Terminal 2: Start frontend
cd frontend && npm install && npm run dev
```

## 🆘 Troubleshooting

### Common Issues:

1. **Model download takes time**: First run downloads AI models (~500MB)
2. **CUDA errors**: System falls back to CPU automatically
3. **Port conflicts**: Change ports in configuration files
4. **Memory issues**: Reduce batch size or use CPU mode

### Quick Fixes:

```bash
# Clear cache and restart
rm -rf .cache/
python test_system.py

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## 📊 System Requirements

- **Minimum**: Python 3.8+, 4GB RAM, 2GB disk space
- **Recommended**: Python 3.9+, 8GB RAM, GPU (optional)
- **Internet**: Required for initial model download

## 🎉 You're Ready!

The system should now be running at `http://localhost:8501`. Start asking questions about the IISc M.Mgt program!

---

**Need help?** Check the main [README.md](README.md) for detailed documentation.
