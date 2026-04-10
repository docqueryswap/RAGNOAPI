import logging
from flask import Flask, render_template, request, jsonify
import os, uuid
from threading import Thread
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

# Global variables with status tracking
vector_db = None
processing_status = {}  # Track processing status by doc_id

def process_file_background(file_path, filename, doc_id):
    global vector_db, processing_status
    try:
        processing_status[doc_id] = 'processing'
        logging.info(f'Background processing started for {filename}')
        
        text = doc_proc.process_uploaded_file(file_path)
        chunks = text_proc.split_text(text)
        
        embeddings = []
        for chunk in chunks:
            emb = text_proc.generate_single_embedding(chunk)
            embeddings.append(emb)
        
        if vector_db is None:
            vector_db = PineconeVectorStore()
        
        vector_db.store_documents(chunks, embeddings, {'doc_id': doc_id})
        processing_status[doc_id] = 'ready'
        logging.info(f'Background processing complete for {filename}')
    except Exception as e:
        processing_status[doc_id] = f'error: {str(e)}'
        logging.error(f'Background processing error: {str(e)}')

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
        doc_id = str(uuid.uuid4())
        
        Thread(target=process_file_background, args=(file_path, filename, doc_id), daemon=True).start()
        
        return jsonify({
            'message': 'File uploaded! Processing in background...',
            'doc_id': doc_id,
            'status': 'processing'
        })
    except Exception as e:
        logging.error(f'Upload error: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/status/<doc_id>', methods=['GET'])
def check_status(doc_id):
    status = processing_status.get(doc_id, 'unknown')
    return jsonify({'doc_id': doc_id, 'status': status})

@app.route('/ask', methods=['POST'])
def ask():
    global vector_db
    try:
        data = request.get_json()
        question = data['question']
        style = data.get('style', 'default')
        doc_id = data.get('doc_id')

        # Check if vector_db is initialized
        if vector_db is None:
            return jsonify({'error': 'No documents have been processed yet. Please upload a document first.'}), 400
        
        # Check if specific document is ready
        if doc_id and processing_status.get(doc_id) != 'ready':
            return jsonify({'error': f'Document is still processing. Current status: {processing_status.get(doc_id, "unknown")}'}), 400

        q_embed = text_proc.generate_query_embedding(question)
        docs = vector_db.search_similar(q_embed)
        context = '\n'.join([d['metadata']['text'] for d in docs])

        prompt = mcp.get_context_prompt(style, question, context)
        answer = rag.generate_answer(prompt)

        return jsonify({'answer': answer})
    except Exception as e:
        logging.error(f'Ask error: {str(e)}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    text_proc._load_model()
    app.run(host='0.0.0.0', port=5000, debug=False)
