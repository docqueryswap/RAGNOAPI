from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextProcessor:
    def __init__(self):
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            # Tiny model: 17MB, works on 512MB Render
            self._model = SentenceTransformer("TaylorAI/gte-tiny")
        return self._model

    def split_text(self, text):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,  # Smaller chunks = less memory
            chunk_overlap=30
        )
        return splitter.split_text(text)

    def generate_embeddings(self, chunks):
        # Process in smaller batches
        return self.model.encode(chunks, normalize_embeddings=True, batch_size=8)

    def generate_query_embedding(self, query):
        return self.model.encode(query, normalize_embeddings=True)
