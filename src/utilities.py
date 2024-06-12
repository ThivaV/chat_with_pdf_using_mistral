"""coversation with mistral ai"""

import PyPDF2
import faiss
from mistralai.client import MistralClient
import numpy as np
import streamlit as st

@st.cache_resource
class Conversation:
    def __init__(self, api_key: str, pdfs_bytes: list) -> None:
        self.api_key = api_key
        self.pdfs_bytes = pdfs_bytes
        self.client = MistralClient(api_key = self.api_key)

        self.chunks = None
        self.vector_db = None

    def extract_embeddings(self, chunk):
        """extract embeddings"""

        embeddings_batch_response = self.client.embeddings(
            model="mistral-embed",
            input=chunk
        )

        return embeddings_batch_response.data[0].embedding
    
    def initialize_retriever(self):
        """initialize retriever"""
        
        if self.pdfs_bytes:
            pdfs = []
            for pdf in self.pdfs_bytes:
                reader = PyPDF2.PdfReader(pdf)
                
                txt = ""
                for page in reader.pages:
                    txt += "\n" + page.extract_text()                
                pdfs.append(txt)
            
            chunk_size = 4096
            chunks = []

            for pdf in pdfs:
                chunks += [pdf[i:i + chunk_size] for i in range(0, len(pdf), chunk_size)]

            self.chunks = chunks

            text_embeddings = np.array([self.extract_embeddings(chunk) for chunk in chunks])
            d = text_embeddings.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(text_embeddings)

            self.vector_db = index
        else:
            self.chunks = None
            self.vector_db = None

    def search(self, query: str):
        """search"""

        if all([self.chunks, self.vector_db]):
            # if both self.chunks & self.vector_db variables are not None
            query_embedding = np.array([self.extract_embeddings(query)])
            D, I = self.vector_db.search(query_embedding, k = 4)
            retrieved_chunk = [self.chunks[i] for i in I.tolist()[0]]
            retrieved_context = "\n\n".join(retrieved_chunk)

            return retrieved_context
        else:
            return []
   
    def talk_to_mistral_ai(self, message):
        """talk to mistral ai_"""

        response = self.client.chat(model= "open-mistral-7b", messages=message, max_tokens=1024)
        return response