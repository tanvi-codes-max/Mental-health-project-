# 🧠 MindGuard AI — Mental Health Awareness & Crisis Support

> **Agentic AI chatbot** powered by **IBM Granite** via watsonx.ai  
> Flask · Python · 5 Specialized Agents · Lightweight RAG · Crisis Detection

---

## Features

| Feature | Details |
|---|---|
| **5 AI Agents** | Empathy, Crisis Intervention, Risk Assessment, Coping Strategies, Wellness Educator |
| **Lightweight RAG** | In-memory TF-IDF knowledge base (12 mental health topics) |
| **Crisis Detection** | Rule-based fast-path before any LLM call |
| **Chat UI** | Single-page dark-mode frontend, no external dependencies |
| **REST API** | `/api/chat`, `/api/history`, `/api/clear`, `/api/health` |

---

## Quick Start

### 1. Clone & set up environment

```bash
git clone <your-repo-url>
cd mental-health-bot
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp .env.example .env
# Edit .env and fill in your IBM watsonx.ai API key and project ID
```

### 3. Run

```bash
python app.py
# Open http://localhost:5000
```

### Production (gunicorn)

```bash
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```

---

## Project Structure

```
mental-health-bot/
├── app.py            # Main application (agents, RAG, Flask routes, UI)
├── requirements.txt  # Python dependencies
├── .env.example      # Environment variable template
└── README.md
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Chat UI |
| `POST` | `/api/chat` | Send message `{"message": "..."}` |
| `GET` | `/api/history` | Get session conversation history |
| `POST` | `/api/clear` | Clear session history |
| `GET` | `/api/health` | Health check |

---

## Crisis Resources

If you or someone you know is in crisis:
- **988** — Suicide & Crisis Lifeline (US, 24/7)
- **Text HOME to 741741** — Crisis Text Line
- **findahelpline.com** — International directory

---

## Disclaimer

MindGuard AI is an **educational and support tool only**. It is **not a substitute** for
professional mental health care. Always seek help from a qualified professional for
diagnosis or treatment.
