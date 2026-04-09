from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

class TextProcessor:
    def __init__(self):
        # BAAI/bge-small: Top-tier small embedding model for RAG
        # Size: ~33MB, fits in Render's 512MB free tier
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    def split_text(self, text):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )
        return splitter.split_text(text)

    def generate_embeddings(self, chunks):
        return self.model.encode(chunks, normalize_embeddings=True)

    def generate_query_embedding(self, query):
        # BGE models need this prefix for optimal query encoding
        return self.model.encode(
            "Represent this sentence for searching relevant passages: " + query,
            normalize_embeddings=True
        )
