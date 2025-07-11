import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

class HatcheryAgent:
    def __init__(self, knowledge_base_path: str):
        self.kb_path = knowledge_base_path
        self.vector_db = None
        self.qa_chain = None
        
        # 1. Setup Embedding Model (Local & Fast)
        print(" Loading Embedding Model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # 2. Setup LLM (Ollama Llama3)
        print(" Connecting to Ollama (Llama 3)...")
        self.llm = Ollama(model="llama3")

    def ingest_knowledge(self):
        """Reads the manual and builds the vector index."""
        print(f" Ingesting Manual: {self.kb_path}")
        
        # Load Data
        loader = TextLoader(self.kb_path)
        documents = loader.load()
        
        # Split into chunks (Paragraphs)
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        
        # Create Vector Store (ChromaDB)
        self.vector_db = Chroma.from_documents(
            documents=texts, 
            embedding=self.embeddings,
            collection_name="hatchery_rules"
        )
        
        # Create the Retrieval Chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_db.as_retriever()
        )
        print(" Knowledge Base Ready!")

    def analyze_defect(self, prediction_class: str) -> str:
        """
        Queries the manual based on the prediction.
        """
        if prediction_class == "fertile":
            return " Egg is Fertile. Proceed to incubation."
            
        # If defect, ask the 'Why' and 'Action'
        query = (
            f"The system detected a '{prediction_class}' egg. "
            "Based on the Hatchery Operation Manual, what is the specific criteria for this defect "
            "and what is the required ACTION?"
        )
        
        print(f"❓ Asking Agent: {query}")
        response = self.qa_chain.invoke(query)
        return response['result']

# Simple test block
if __name__ == "__main__":
    # Point to the dummy manual we created
    kb_path = "data/knowledge_base/manual.txt"
    
    agent = HatcheryAgent(kb_path)
    agent.ingest_knowledge()
    
    # Simulate a "Defect" detection
    print("\n--- TEST REPORT ---")
    report = agent.analyze_defect("defect")
    print(report)