# shopify

Overview


The Shopkeeper Assistant Chatbot is an intelligent retrieval-based chatbot designed to assist users by answering queries about products from a predefined dataset of CSV files. Built using Streamlit for UI, LangChain for conversational modeling, FAISS for vector storage, and Google Generative AI for embeddings, the chatbot focuses on enhancing user interaction through natural language understanding and retrieval-augmented generation (RAG).

Key Features


1.Document Loading and Embedding:

Uses GoogleGenerativeAIEmbeddings to generate embeddings for textual data.
Supports document similarity search to retrieve relevant product information.


2.Contextual Responses:

Applies a template-based prompt system to maintain context and ensure relevant responses based solely on loaded CSV data.
Query Classification:

Recognizes and classifies user queries to provide more targeted responses, such as price inquiries, availability checks, and general recommendations.


3.Conversation History:

Displays the conversation history between the user and chatbot for a consistent dialogue flow.
