import os
import pickle
from dotenv import load_dotenv
from filetype import guess
from langchain.document_loaders import UnstructuredImageLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

load_dotenv() # Load the .env file

class PDF_AI:
    def __init__(self, file_path):
        self.key = os.getenv("OPENAI_API_KEY")
        self.file_path = file_path
        self._file_type = None
        self._file_content = None
        self._file_splitter = None
        self._embeddings = None
        self._document_search = None
        self._chain = None

    @property
    def file_type(self):
        if not self._file_type:
            self._file_type = self.detect_document_type()
        return self._file_type
    
    @property
    def file_content(self):
        if not self._file_content:
            self._file_content = self.extract_file_content()
        return self._file_content
    
    @property
    def file_splitter(self):
        if not self._file_splitter:
            try:
                self._file_splitter = self.load_chunks_from_file("chunks_cache.pkl")
            except (FileNotFoundError, pickle.UnpicklingError):
                self._file_splitter = self.create_chunks().split_text(self.file_content)
                self.save_chunks_to_file(self._file_splitter, "chunks_cache.pkl")
        return self._file_splitter
    
    @property
    def embeddings(self):
        if not self._embeddings:
            self._embeddings = self.create_embeddings()
        return self._embeddings

    @property
    def document_search(self):
        if not self._document_search:
            self._document_search = self.get_doc_search(self.file_splitter, self.embeddings)
        return self._document_search

    @property
    def chain(self):
        if not self._chain:
            self._chain = self.create_chain()
        return self._chain
        
    def detect_document_type(self):
        guess_file = guess(self.file_path)
        file_type = ""
        image_types = ['jpg', 'jpeg', 'png', 'gif']
        if(guess_file.extension.lower() == "pdf"):
            file_type = "pdf"
        elif(guess_file.extension.lower() in image_types):
            file_type = "image"
        else:
            file_type = "unknown"
        return file_type
    
    def extract_file_content(self):
        if(self.file_type == "pdf"):
            loader = UnstructuredFileLoader(self.file_path)
        elif(self.file_type == "image"):
            loader = UnstructuredImageLoader(self.file_path)
        documents = loader.load()
        documents_content = '\n'.join(doc.page_content for doc in documents)
        return documents_content
    
    def create_chunks(self):
        text_splitter = CharacterTextSplitter(        
            separator = "\n\n",
            chunk_size = 1000,
            chunk_overlap  = 200,
            length_function = len,
        )
        return text_splitter
    
    def create_embeddings(self):
        embeddings = OpenAIEmbeddings()
        return embeddings
    
    def get_doc_search(self, file_chunks, embeddings):
        return FAISS.from_texts(file_chunks, embeddings)
    
    def create_chain(self):
        return load_qa_chain(OpenAI(), chain_type = "map_rerank",  
                      return_intermediate_steps=True)
    
    def save_chunks_to_file(self, chunks, filename):
        with open(filename, 'wb') as file:
            pickle.dump(chunks, file)

    def load_chunks_from_file(self, filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
    
    def chat_with_file(self, query):
        # Attempt to load cached chunks
        try:
            file_splitter = self.load_chunks_from_file("chunks_cache.pkl")
        except (FileNotFoundError, pickle.UnpicklingError):
            file_content = self.extract_file_content()
            file_splitter = self.create_chunks().split_text(file_content)
            # Save the chunks to a file for future use
            self.save_chunks_to_file(file_splitter, "chunks_cache.pkl")

        embeddings = self.create_embeddings()
        document_search = self.get_doc_search(file_splitter, embeddings)
        documents = document_search.similarity_search(query)
        results = chain({
                            "input_documents": documents,
                            "question": query
                        }, 
                        return_only_outputs=True)
        results = results['intermediate_steps'][0]
        return results


if __name__ == "__main__":
    file_path = "pentagon_on_prc.pdf"
    #Detect Document Type
    PDF_AI = PDF_AI(file_path)
    #Chat with PDF File:
    chain = PDF_AI.create_chain()
    query = "What are the key takeaways of this paper as per the pentagon?"
    results = PDF_AI.chat_with_file(query)
    answer = results["answer"]
    confidence_score = results["score"]
    print(f"Confidence Score: {confidence_score}")