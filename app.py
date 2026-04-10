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

# ✅ FIXED: No global state - stateless access
def get_vector_db():
    return PineconeVectorStore()

def process_file_background(file_path, filename, doc_id, namespace):
    try:
        logging.info(f'Background processing started for {filename}')
        
        text = doc_proc.process_uploaded_file(file_path)
        chunks = text_proc.split_text(text)
        
        embeddings = []
        for chunk in chunks:
            emb = text_proc.generate_single_embedding(chunk)
            embeddings.append(emb)
        
        # ✅ Fresh connection, with namespace
        vdb = PineconeVectorStore()
        vdb.store_documents(chunks, embeddings, {'doc_id': doc_id}, namespace=namespace)
        
        logging.info(f'Background processing complete for {filename}')
    except Exception as e:
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
        namespace = f"doc_{doc_id}"
        
        # ✅ Pass fresh connection to thread
        Thread(target=process_file_background, args=(file_path, filename, doc_id, namespace), daemon=True).start()
        
        return jsonify({
            'message': 'File uploaded! Processing in background...',
            'doc_id': doc_id,
            'namespace': namespace,
            'status': 'processing'
        })
    except Exception as e:
        logging.error(f'Upload error: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        question = data['question']
        style = data.get('style', 'default')
        namespace = data.get('namespace', 'doc_default')

        # ✅ Fresh connection every request
        vdb = get_vector_db()
        
        # ✅ Check if namespace has data
        stats = vdb.get_index_stats()
        if namespace not in stats.get('namespaces', {}):
            return jsonify({'error': 'No documents found. Please upload a document first.'}), 400

        q_embed = text_proc.generate_query_embedding(question)
        docs = vdb.search_similar(q_embed, namespace=namespace)
        
        if not docs:
            return jsonify({'error': 'No relevant content found in document.'}), 400
            
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
