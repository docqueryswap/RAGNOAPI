from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self):
        # ✅ LOAD MODEL IMMEDIATELY - No lazy loading!
        logger.info("Loading SentenceTransformer model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        logger.info("Model loaded successfully!")

    def split_text(self, text):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=30
        )
        return splitter.split_text(text)

    def generate_single_embedding(self, chunk):
        # ✅ Model guaranteed to be loaded
        return self.model.encode(chunk, normalize_embeddings=True)

    def generate_query_embedding(self, query):
        # ✅ Model guaranteed to be loaded
        return self.model.encode(query, normalize_embeddings=True)
