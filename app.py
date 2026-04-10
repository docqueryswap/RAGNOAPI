import logging
import time
from flask import Flask, render_template, request, jsonify
import os, uuid
from threading import Thread, Lock
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

# Use a lock for thread-safe initialization
vector_db = None
vector_db_lock = Lock()
processing_status = {}

def get_vector_db():
    global vector_db
    with vector_db_lock:
        if vector_db is None:
            logging.info('Initializing Pinecone connection...')
            vector_db = PineconeVectorStore()
            logging.info('Pinecone connected!')
    return vector_db

def process_file_background(file_path, filename, doc_id):
    global processing_status
    try:
        processing_status[doc_id] = 'processing'
        logging.info(f'Background processing started for {filename}')
        
        text = doc_proc.process_uploaded_file(file_path)
        chunks = text_proc.split_text(text)
        
        embeddings = []
        for chunk in chunks:
            emb = text_proc.generate_single_embedding(chunk)
            embeddings.append(emb)
        
        # Get or create vector_db
        vdb = get_vector_db()
        vdb.store_documents(chunks, embeddings, {'doc_id': doc_id})
        
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
        
        # Initialize vector_db in main thread before background processing
        get_vector_db()
        
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
    try:
        data = request.get_json()
        question = data['question']
        style = data.get('style', 'default')

        # Get vector_db (will initialize if needed)
        vdb = get_vector_db()
        
        # Check if we have any documents at all
        try:
            # Simple query to check if index has data
            test_embed = text_proc.generate_query_embedding("test")
            docs = vdb.search_similar(test_embed)
            
            if not docs or len(docs) == 0:
                return jsonify({'error': 'No documents found in database. Please upload a document first.'}), 400
        except Exception as e:
            logging.error(f'Index check error: {str(e)}')
            return jsonify({'error': 'Database connection issue. Please try again in a moment.'}), 500

        q_embed = text_proc.generate_query_embedding(question)
        docs = vdb.search_similar(q_embed)
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
