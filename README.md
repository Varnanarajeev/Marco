# Marco

Overview


The Shopkeeper Assistant Chatbot is an RAG based chatbot designed to assist users by answering queries about products from a predefined dataset of CSV file provided by the user. Built using Streamlit for UI, LangChain for conversational modeling, FAISS for vector storage, and Google Generative AI for embeddings, the chatbot focuses on enhancing user interaction through natural language understanding and retrieval-augmented generation (RAG).

Key Features


1.Document Loading and Embedding:

Uses GoogleGenerativeAIEmbeddings to generate embeddings for textual data.
Supports document similarity search to retrieve relevant product information.


2.Contextual Responses:

Applies a template-based prompt system to maintain context and ensure relevant responses based solely on loaded CSV data.


3.Query Classification:

Recognizes and classifies user queries to provide more targeted responses, such as price inquiries, availability checks, and general recommendations.


4.Conversation History:

Displays the conversation history between the user and chatbot for a consistent dialogue flow.


Approach


1. Data Loading and Preparation

The bot starts by loading CSV files containing product data, stored in a specified directory. Each product entry is extracted as a document and then converted into embeddings using Google Generative AI embeddings, making it easier to identify similar products based on user queries.

CSVLoader: Reads CSV files from a specified directory and parses them into documents, each holding the product’s description, price, category, stock, and supplier information.

Recursive Text Splitting: Each document is split into manageable chunks (default chunk size: 500 characters) with slight overlaps (100 characters) to preserve context while ensuring that each chunk is processable.


2. Vector Database Choice: FAISS

   
The FAISS (Facebook AI Similarity Search) vector database is used to store and retrieve product embeddings for efficient similarity searches. FAISS is well-suited for handling high-dimensional data due to its efficient indexing and retrieval capabilities.

3. Model Selection

ChatGroq: The primary LLM (Large Language Model) in use is ChatGroq’s "Llama3-8b-8192," chosen for its balance between accuracy, context retention, and performance.

4.GoogleGenerativeAIEmbeddings: These embeddings provide a robust way to represent textual product information, facilitating accurate retrieval and enabling contextual responses.


5. Prompt Template Design


Using LangChain’s ChatPromptTemplate, a templated prompt is defined to instruct the model to respond based on the CSV context only. The prompt is structured as follows:


Installation

Clone Repository: Clone this repository to your local machine.


Key Design Decisions


Use of FAISS for Vector Store: FAISS was selected for its efficient handling of large volumes of vectors, which are crucial in applications requiring rapid similarity search.

Embedding Model Choice: Google Generative AI embeddings were chosen for their high accuracy, contextual depth, and API accessibility, ensuring robust document retrieval.

Streamlit for UI: Streamlit offers a quick and user-friendly interface that can be adapted for real-time interactions, making it ideal for a chatbot application
