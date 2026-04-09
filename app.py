import logging
from flask import Flask, render_template, request, jsonify
import os, uuid
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

from document_processor import DocumentProcessor
from text_processor import TextProcessor
from vector_store import PineconeVectorStore
from mcp_protocol import ModelContextProtocol
from rag_pipeline import RAGPipeline

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

doc_proc = DocumentProcessor()
text_proc = TextProcessor()
rag = RAGPipeline()
mcp = ModelContextProtocol()
vector_db = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        file = request.files['file']
        filename = file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        text = doc_proc.process_uploaded_file(file_path)
        chunks = text_proc.split_text(text)
        embeddings = text_proc.generate_embeddings(chunks)

        global vector_db
        if vector_db is None:
            vector_db = PineconeVectorStore()

        doc_id = str(uuid.uuid4())
        vector_db.store_documents(chunks, embeddings, {'doc_id': doc_id})

        return jsonify({'message': 'File processed successfully!', 'doc_id': doc_id})
    except Exception as e:
        logging.error(f'Upload error: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        question = data['question']
        style = data.get('style', 'default')

        q_embed = text_proc.generate_embeddings([question])[0]
        docs = vector_db.search_similar(q_embed)
        context = '\n'.join([d['metadata']['text'] for d in docs])

        prompt = mcp.get_context_prompt(style, question, context)
        answer = rag.generate_answer(prompt)

        return jsonify({'answer': answer})
    except Exception as e:
        logging.error(f'Ask error: {str(e)}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)