# ChatYourDocs x RAG

## Introduction
ChatYourDocs x RAG is an application that leverages Retrieval Augmented Generation (RAG) to query a vector database instead of using a relational database to store embeddings. It combines the power of language models with efficient vector retrieval to provide detailed responses based on document content.

## Prerequisites
- Python 3.8 or later
- Required Python packages (install using `pip install`):
  - streamlit
  - langchain
  - PyPDF2
  - dotenv
  - openai

## Getting Started
1. Clone this repository to your local machine.

2. Create a `.env` file in the project directory and set your OpenAI API key as follows:

3. Install the required Python packages if not already installed:

4. Run the application using Streamlit:

## Usage
- Upload PDF, DOC, DOCX, or TXT files containing documents.
- Ask questions related to the uploaded documents.
- The application will use RAG and a vector database to generate detailed responses based on document content.

## File Descriptions
- `app.py`: The main application script that powers the user interface and functionalities.
- `functions.py`: Core functions responsible for database connections, document processing, and query answering.

## Function Descriptions
- **Vector Database Management**: Uses FAISS local vector database. Functions to initialize the database and manage conversation histories.
- **Document Processing**: Functions to read and process PDF, DOCX, and TXT files, and store document content and embeddings in the vector database.
- **Query Answering**: Utilizes RAG and the vector database to generate responses to user queries based on document content.

## Conversation History
- The application maintains a conversation history that can be cleared using the "Clear Conversation History" button.

## Contributions
Contributions to enhance the application's functionalities are welcome. Feel free to fork the repository and open pull requests.

## License
This application is available under the MIT License. See the [LICENSE](LICENSE) file for more details.

