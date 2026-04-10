import logging
import sys
import traceback
from flask import Flask, render_template, request, jsonify
import os, uuid
import tempfile
from dotenv import load_dotenv

load_dotenv()

# ✅ CRITICAL: Enable detailed logging for Render debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

from document_processor import DocumentProcessor
from text_processor import TextProcessor
from vector_store import PineconeVectorStore
from mcp_protocol import ModelContextProtocol
from rag_pipeline import RAGPipeline

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

# ✅ File size limit for Render free tier (5MB)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

doc_proc = DocumentProcessor()
text_proc = TextProcessor()
rag = RAGPipeline()
mcp = ModelContextProtocol()

NAMESPACE = "rag-docs"

def get_vector_db():
    return PineconeVectorStore()

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Please upload under 5MB.'}), 413

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    logger.info("=" * 50)
    logger.info("UPLOAD REQUEST RECEIVED")
    
    try:
        # ✅ STEP 1: Check file exists
        if 'file' not in request.files:
            logger.error("No file in request")
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        filename = file.filename
        logger.info(f"STEP 1: File received - {filename}")
        
        # ✅ STEP 2: Use tempfile (Render-safe)
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
            file.save(tmp.name)
            file_path = tmp.name
            logger.info(f"STEP 2: File saved to temp path - {file_path}")
        
        doc_id = str(uuid.uuid4())
        logger.info(f"STEP 3: Generated doc_id - {doc_id}")
        
        # ✅ STEP 4: Process SYNCHRONOUSLY (no threads)
        logger.info("STEP 4: Starting document processing...")
        
        # Extract text
        logger.info("  → Extracting text from document...")
        text = doc_proc.process_uploaded_file(file_path)
        logger.info(f"  → Extracted {len(text)} characters")
        
        # Split into chunks
        logger.info("  → Splitting into chunks...")
        chunks = text_proc.split_text(text)
        logger.info(f"  → Created {len(chunks)} chunks")
        
        # Generate embeddings
        logger.info("  → Generating embeddings...")
        embeddings = []
        for i, chunk in enumerate(chunks):
            if i % 10 == 0:
                logger.info(f"  → Processing chunk {i+1}/{len(chunks)}")
            emb = text_proc.generate_single_embedding(chunk)
            embeddings.append(emb)
        logger.info(f"  → Generated {len(embeddings)} embeddings")
        
        # Store in Pinecone
        logger.info("  → Storing in Pinecone...")
        vdb = PineconeVectorStore()
        vdb.store_documents(chunks, embeddings, {'doc_id': doc_id}, namespace=NAMESPACE)
        logger.info("  → Stored successfully!")
        
        # ✅ STEP 5: Verify storage
        logger.info("STEP 5: Verifying Pinecone storage...")
        stats = vdb.get_index_stats()
        logger.info(f"  → Index stats: {stats}")
        
        # Clean up temp file
        os.unlink(file_path)
        logger.info("STEP 6: Temp file cleaned up")
        
        logger.info("✅ UPLOAD COMPLETED SUCCESSFULLY")
        logger.info("=" * 50)
        
        return jsonify({
            'message': 'File processed successfully!',
            'doc_id': doc_id,
            'chunks': len(chunks),
            'status': 'ready'
        })
        
    except Exception as e:
        logger.error("=" * 50)
        logger.error("❌ PROCESSING FAILED")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("Traceback:")
        logger.error(traceback.format_exc())
        logger.error("=" * 50)
        
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/ask', methods=['POST'])
def ask():
    logger.info("QUERY REQUEST RECEIVED")
    
    try:
        data = request.get_json()
        question = data['question']
        style = data.get('style', 'default')
        logger.info(f"Question: {question[:50]}...")

        vdb = get_vector_db()
        
        # Check Pinecone directly
        stats = vdb.get_index_stats()
        logger.info(f"Index stats: {stats}")
        
        if NAMESPACE not in stats.get('namespaces', {}):
            logger.warning(f"Namespace '{NAMESPACE}' not found in index")
            return jsonify({'error': 'No documents found. Please upload a document first.'}), 400

        q_embed = text_proc.generate_query_embedding(question)
        docs = vdb.search_similar(q_embed, namespace=NAMESPACE)
        
        if not docs:
            return jsonify({'error': 'No relevant content found.'}), 400
            
        context = '\n'.join([d['metadata']['text'] for d in docs])

        prompt = mcp.get_context_prompt(style, question, context)
        answer = rag.generate_answer(prompt)
        
        logger.info("Query answered successfully")

        return jsonify({'answer': answer})
    except Exception as e:
        logger.error(f'Ask error: {str(e)}')
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting application...")
    logger.info("Loading embedding model...")
    text_proc._load_model()
    logger.info("Model loaded successfully!")
    logger.info("Application ready!")
    app.run(host='0.0.0.0', port=5000, debug=False)
