# SQL Server RAG Application

This application implements a Retrieval-Augmented Generation (RAG) system that connects to a SQL Server database and provides answers based on the data stored in it.

## Prerequisites

- Python 3.8 or higher
- SQL Server
- OpenAI API key
- ODBC Driver for SQL Server

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install the SQL Server ODBC driver for your operating system

## Configuration

1. Copy the `.env.example` file to `.env`
2. Update the `.env` file with your configuration:
   - Database connection details
   - OpenAI API key
   - Vector store path

## Usage

1. Update the `table_name` variable in `app.py` with your target table name
2. Run the application:
   ```bash
   python app.py
   ```

The application will:
1. Connect to your SQL Server database
2. Create a vector store from your table data
3. Allow you to query the data using natural language

## How it Works

1. The application connects to your SQL Server database
2. It fetches data from the specified table
3. The data is converted into text chunks and embedded using sentence transformers
4. A vector store is created using Chroma
5. When you ask a question, the system:
   - Retrieves relevant context from the vector store
   - Uses OpenAI's GPT model to generate an answer based on the context
   - Returns the answer along with the source documents

## Customization

You can customize the following aspects:
- Embedding model by changing the `model_name` in the `HuggingFaceEmbeddings` initialization
- Chunk size and overlap in the `RecursiveCharacterTextSplitter`
- Prompt template in the `get_qa_chain` method
- LLM model and parameters in the `ChatOpenAI` initialization

## License

MIT 