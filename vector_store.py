import os
import time
import logging
from pinecone import Pinecone, ServerlessSpec

class PineconeVectorStore:
    def __init__(self):
        self.index_name = os.getenv('PINECONE_INDEX_NAME', 'ragnoapi-384')
        self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        
        if self.index_name not in self.pc.list_indexes().names():
            logging.info('Creating index ' + self.index_name + '...')
            self.pc.create_index(
                name=self.index_name,
                dimension=384,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
            while not self.pc.describe_index(self.index_name).status.get('ready', False):
                time.sleep(2)
        self.index = self.pc.Index(self.index_name)

    def store_documents(self, chunks, vectors, metadata, namespace="rag-docs"):
        records = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            records.append({
                'id': metadata['doc_id'] + '-' + str(i),
                'values': vector.tolist(),
                'metadata': {'text': chunk, **metadata}
            })
        self.index.upsert(vectors=records, namespace=namespace)
        logging.info(f'Stored {len(records)} vectors in namespace: {namespace}')

    def search_similar(self, query_vector, top_k=3, namespace="rag-docs"):
        result = self.index.query(
            vector=query_vector.tolist(),
            top_k=top_k,
            include_metadata=True,
            namespace=namespace,
            timeout=10
        )
        return result.matches

    def get_index_stats(self):
        stats = self.index.describe_index_stats()
        logging.info(f'Index stats: {stats}')
        return stats
