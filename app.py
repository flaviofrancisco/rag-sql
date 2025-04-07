import os
from dotenv import load_dotenv
import pyodbc
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
# Load environment variables
load_dotenv()

import transformers
print(transformers.__version__)

class SQLRAG:
    def __init__(self):
        # Database connection using Windows Authentication with basic parameters
        conn_str = (
            f"DRIVER={{ODBC Driver 18 for SQL Server}};"
            f"SERVER={os.getenv('DB_SERVER')};"
            f"DATABASE={os.getenv('DB_DATABASE')};"
            "Trusted_Connection=yes;"
            "TrustServerCertificate=yes;"
        )
        
        print(f"Attempting to connect with connection string: {conn_str}")
        self.conn = pyodbc.connect(conn_str)
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # Initialize vector store
        self.vector_store_path = os.getenv('VECTOR_STORE_PATH')
        self.vector_store = None
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0
        )
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def fetch_data_from_sql(self, query):
        """Fetch data from SQL Server"""
        cursor = self.conn.cursor()
        cursor.execute(query)
        columns = [column[0] for column in cursor.description]
        results = []
        
        for row in cursor.fetchall():
            results.append(dict(zip(columns, row)))
            
        return results

    def create_vector_store(self, table_name):
        """Create vector store from SQL data"""
        # Fetch data from SQL
        query = f"SELECT * FROM {table_name}"
        data = self.fetch_data_from_sql(query)
        
        # Convert data to text
        texts = []
        for row in data:
            text = " ".join([f"{k}: {v}" for k, v in row.items()])
            texts.append(text)
        
        # Split texts into chunks
        chunks = self.text_splitter.create_documents(texts)
        
        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.vector_store_path
        )
        
        # Save vector store
        self.vector_store.persist()

    def load_vector_store(self):
        """Load existing vector store"""
        if os.path.exists(self.vector_store_path):
            self.vector_store = Chroma(
                persist_directory=self.vector_store_path,
                embedding_function=self.embeddings
            )

    def get_qa_chain(self):
        """Create QA chain"""
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}

        Question: {question}

        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

    def query(self, question):
        """Query the RAG system"""
        if not self.vector_store:
            raise Exception("Vector store not initialized. Please create or load a vector store first.")
        
        qa_chain = self.get_qa_chain()
        result = qa_chain({"query": question})
        return result

def main():
    # Initialize RAG system
    rag = SQLRAG()
    
    # Example usage
    table_name = "_attributes"  # Replace with your table name
    rag.create_vector_store(table_name)


    while True:
        # creata code where the user can ask questions
        question = input("Enter a question: ")
        result = rag.query(question)
        print(f"Question: {question}")
        print(f"Answer: {result['result']}")
        print("\nSources:")
        for doc in result['source_documents']:
            print(f"- {doc.page_content[:100]}...")
    



if __name__ == "__main__":
    main()
