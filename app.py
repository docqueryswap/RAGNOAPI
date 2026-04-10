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

# ✅ FIXED: Single consistent namespace
NAMESPACE = "rag-docs"

def get_vector_db():
    return PineconeVectorStore()

def process_file_sync(file_path, filename, doc_id):
    """Process file synchronously - reliable on Render"""
    try:
        logging.info(f'Processing started for {filename}')
        
        text = doc_proc.process_uploaded_file(file_path)
        chunks = text_proc.split_text(text)
        
        embeddings = []
        for chunk in chunks:
            emb = text_proc.generate_single_embedding(chunk)
            embeddings.append(emb)
        
        vdb = PineconeVectorStore()
        vdb.store_documents(chunks, embeddings, {'doc_id': doc_id}, namespace=NAMESPACE)
        
        logging.info(f'Processing complete for {filename}')
        return True
    except Exception as e:
        logging.error(f'Processing error: {str(e)}')
        return False

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
        
        # ✅ Process synchronously - no unreliable threads on Render
        success = process_file_sync(file_path, filename, doc_id)
        
        if success:
            return jsonify({
                'message': 'File processed successfully!',
                'doc_id': doc_id,
                'status': 'ready'
            })
        else:
            return jsonify({'error': 'Processing failed'}), 500
            
    except Exception as e:
        logging.error(f'Upload error: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        question = data['question']
        style = data.get('style', 'default')

        # ✅ Fresh connection every request
        vdb = get_vector_db()
        
        # ✅ Check Pinecone directly - source of truth
        stats = vdb.get_index_stats()
        
        if NAMESPACE not in stats.get('namespaces', {}):
            return jsonify({'error': 'No documents found in database. Please upload a document first.'}), 400

        q_embed = text_proc.generate_query_embedding(question)
        docs = vdb.search_similar(q_embed, namespace=NAMESPACE)
        
        if not docs:
            return jsonify({'error': 'No relevant content found.'}), 400
            
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
