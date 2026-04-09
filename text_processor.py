from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

class TextProcessor:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def split_text(self, text):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        return splitter.split_text(text)

    def generate_embeddings(self, chunks):
        return self.model.encode(chunks)