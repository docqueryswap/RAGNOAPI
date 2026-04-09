from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextProcessor:
    def __init__(self):
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    def split_text(self, text):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        return splitter.split_text(text)

    def generate_embeddings(self, chunks):
        return self.model.encode(chunks, normalize_embeddings=True)

    def generate_query_embedding(self, query):
        return self.model.encode(query, normalize_embeddings=True)
