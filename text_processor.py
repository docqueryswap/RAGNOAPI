from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

class TextProcessor:
    def __init__(self):
        self.model = None

    def _load_model(self):
        # Load model ONCE at startup
        if self.model is None:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def split_text(self, text):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=30
        )
        return splitter.split_text(text)

    def generate_single_embedding(self, chunk):
        # Process ONE chunk at a time to avoid memory spikes
        return self.model.encode(chunk, normalize_embeddings=True)

    def generate_query_embedding(self, query):
        return self.model.encode(query, normalize_embeddings=True)
