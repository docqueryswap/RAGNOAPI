📚 DocuQuery — A RAG Application: The Complete Development Journey
v1.0.0 Production Ready Groq + Llama 3.1

DocuQuery is a Retrieval-Augmented Generation (RAG) application that lets users upload documents (PDF, DOCX, TXT) and ask questions about their content. It uses Pinecone for vector storage, Sentence Transformers for embeddings, and Groq's Llama 3.1 for free, high‑quality answers.

This README documents every step of the development process — from the initial idea to the final working product, including all the errors, dead ends, and breakthroughs.

🧭 Development Timeline & Debugging Odyssey
🔰 Phase 1: Initial Setup & OpenAI (Failed – Cost Barrier)
Started with OpenAI API (gpt-4.1-mini). Worked locally but required prepaid credits. ❌ Abandoned due to cost for free-tier users.
🤖 Phase 2: Hugging Face Inference API (Failed – Reliability Issues)
Switched to free Hugging Face models (google/flan-t5-large, microsoft/phi-2). Encountered 503 Service Unavailable, cold‑start delays, and models echoing questions instead of answering. ❌ Inconsistent quality, unusable for production.
🧪 Phase 3: Local Transformers (Failed – Performance & Quality)
Loaded google/flan-t5-small/base/large locally with transformers. Models either generated gibberish ("a. a. a. a...") or just repeated the question. Inference was slow on CPU (30–60 seconds). ❌ Unacceptable user experience.
🌐 Phase 4: Gemini API (Failed – Quota Exceeded)
Tried Google's Gemini free tier. Received 429 Quota exceeded after a few requests. Free tier has extremely low limits (often 0 RPM). ❌ Not viable for real usage.
📜 Phase 5: Claude API (Failed – No Free Tier)
Attempted Anthropic's Claude. Immediate 429 because free credits require payment method verification. ❌ Not actually free.
🦙 Phase 6: Groq (Success – But Model Deprecation)
Groq's free tier worked beautifully with llama3-8b-8192. However, after a few days, we hit: model_decommissioned. ⚠️ Model was deprecated on Aug 30, 2025.
✅ Phase 7: Groq + Llama 3.1 (Final Working Solution)
Updated to the recommended model llama-3.1-8b-instant. Everything worked flawlessly. Fast (1–2 sec), free, and high‑quality answers. ✅ Production‑ready!

🐞 Detailed Error Log & Fixes
Error	Cause	✅ How We Fixed It
TemplateNotFound: index.html	Missing templates/ folder	Run mkdir -p templates before starting the app
Could not find a version that satisfies the requirement pinecone==8.2.0	Version pin too strict	Use flexible range: pinecone>=8.0,<9.0
SyntaxError: f-string: expecting '}'	Nested quotes inside f‑string	Replaced with string concatenation (+)
Model returns question verbatim	FLAN‑T5 not instruction‑tuned for QA	Switched to Groq's Llama 3.1 (chat model)
Error code: 429 – Quota exceeded (Gemini)	Free tier limits reached	Moved to Groq (generous free tier)
model 'llama3-8b-8192' has been decommissioned	Groq deprecated older Llama 3 model	Updated to llama-3.1-8b-instant

📁 Final Project Structure
docuquery/
├── app.py                 # Flask web server
├── document_processor.py  # PDF/DOCX text extraction
├── text_processor.py      # Chunking & embedding generation
├── vector_store.py        # Pinecone vector DB operations
├── rag_pipeline.py        # Groq LLM integration
├── mcp_protocol.py        # Prompt formatting (styles)
├── requirements.txt       # Python dependencies
├── .env                   # API keys (excluded from Git)
├── .gitignore
└── templates/
    └── index.html         # Frontend UI

⚙️ Installation & Usage
1️⃣ Clone & Setup Environment
git clone https://github.com/yourusername/docuquery.git
cd docuquery
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

2️⃣ Configure Environment Variables
Create a .env file in the root directory:

PINECONE_API_KEY=pcsk_YOUR_KEY_HERE
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=docqueryswap
SECRET_KEY=your-secret-key-here
GROQ_API_KEY=gsk_YOUR_GROQ_KEY_HERE

Get your free Groq API key: console.groq.com (no credit card required).

3️⃣ Run the Application
# Ensure templates folder exists
mkdir -p templates

# Start Flask server
python app.py

Open http://localhost:5000 in your browser.

🧩 Core Components Explained
📄 Document Processing (document_processor.py)
Extracts raw text from uploaded PDF, DOCX, or TXT files using pypdf and python-docx.

✂️ Text Chunking & Embeddings (text_processor.py)
Splits text into overlapping chunks (500 chars, 50 overlap) using LangChain's RecursiveCharacterTextSplitter.
Generates 384‑dimensional embeddings with sentence-transformers/all-MiniLM-L6-v2.

🗄️ Vector Store (vector_store.py)
Uses Pinecone serverless index (AWS us-east-1).
Automatically creates index if it doesn't exist and waits for readiness.
Stores chunks with metadata and performs similarity search.

🤖 LLM Integration (rag_pipeline.py)
Uses Groq's Python SDK to call llama-3.1-8b-instant.
Free tier allows 30 RPM / 14,400 RPD — more than enough for personal use.
Returns concise, context‑aware answers.

🎨 Prompt Styling (mcp_protocol.py)
Applies tone modifiers: default, kid (ELI5), legal, or short.

🌐 Flask App (app.py)
Two main endpoints:
POST /upload – processes document, stores vectors.
POST /ask – retrieves relevant chunks, builds prompt, streams answer from Groq.

📦 Dependencies (requirements.txt)
flask>=3.0,<4.0
langchain>=1.2,<2.0
langchain-text-splitters>=1.0,<2.0
pinecone>=8.0,<9.0
sentence-transformers>=5.0,<6.0
pypdf>=6.0,<7.0
python-docx>=1.1,<2.0
python-dotenv>=1.0,<2.0
groq>=0.9.0
gunicorn>=21.0,<26.0

🚀 Deployment
This app can be deployed easily on Render, Railway, or Heroku. Use gunicorn as the production server:
gunicorn app:app --bind 0.0.0.0:$PORT

Make sure to set the environment variables in your hosting platform.

📝 Lessons Learned
Free ≠ Always Free – Many "free" APIs have hidden quotas or require billing setup. Groq is a rare exception.
Model Selection Matters – Small open‑source models (like FLAN‑T5) often lack the instruction‑following ability needed for RAG.
Error Handling is Crucial – Wrapping endpoints in try/except with logging saved hours of debugging.
Keep Dependencies Flexible – Pinning exact versions leads to conflicts; use compatible release ranges.
Always Check Deprecation Notices – Groq's model deprecation caught us off guard, but the fix was simple.

🤝 Contributing
Found a bug or want to add a feature? PRs are welcome! Please open an issue first to discuss your ideas.

📄 License
MIT License — feel free to use this project for personal or commercial purposes.

Built with ❤️ and a lot of debugging.
DocuQuery — Finally working, thanks to Groq and Llama 3.1.