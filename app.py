import os
import time
import pandas as pd
import streamlit as st
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.schema import Document



# API keys
groq_api_key = "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"
google_api_key = "AIzaSyAOWJGZ7YsJxefdaNzQK8RSfGxExBBa4g0"

# Set environment variable
os.environ["GOOGLE_API_KEY"] = google_api_key

# Streamlit title
st.title("Marco : Shopkeeper Assistant")

# Initialize the language model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

def vector_embedding():
    if "vectors" not in st.session_state:
        model_name = "models/embedding-001"
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model=model_name)
        
        csv_directory = r"C:\talrop_shopify_\data"  # Change this to your CSV directory
        if not os.path.exists(csv_directory):
            st.error("The specified directory does not exist.")
            return
        
        csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]
        if not csv_files:
            st.error("No CSV files found in the specified directory.")
            return
        
        # Load CSV files
        st.session_state.docs = []
        for csv_file in csv_files:
            df = pd.read_csv(os.path.join(csv_directory, csv_file))
            for index, row in df.iterrows():
                content = row.to_string(index=False)
                st.session_state.docs.append(Document(page_content=content, metadata={"source": csv_file}))

        if not st.session_state.docs:
            st.error("No documents loaded. Please check the CSV directory.")
            return
        
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        
        if not st.session_state.final_documents:
            st.error("No documents after splitting. Please check the document content.")
            return
        
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.success("We're ready to go! Shall we start ?")

# Function to classify user queries
def classify_query(user_input):
    # Example simple keyword-based classification (can be enhanced with more complex NLP)
    if "available" in user_input.lower():
        return "availability"
    elif "recommend" in user_input.lower() or "suggest" in user_input.lower():
        return "recommendation"
    elif "price" in user_input.lower() or "cost" in user_input.lower():
        return "details"
    else:
        return "general"

# Button to trigger vector embedding

if st.button("Initialize AI for Assistance"):
    vector_embedding()  # Initialize the embeddings
    
    # Typing effect for the welcome message
    message = "Hey!! I'm Marco, your Shopkeeper Assistant bot. Ask me your queries about product details, stock information, and much more!"
    typing_placeholder = st.empty()  # Create a placeholder for the typing animation
    
    typing_message = ""
    for char in message:
        typing_message += char
        # Use HTML to change the font size
        typing_placeholder.markdown(f"<h5>{typing_message}</h6>", unsafe_allow_html=True)  # Adjust the header size (h5 for smaller text)
        time.sleep(0.02)  # Adjust the delay for typing speed

    
# Input for user question
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

prompt1 = st.text_input("Ask me your queries!")



if prompt1:
    # Check if vectors are initialized before accessing
    if "vectors" not in st.session_state:
        st.error("Vector store not initialized. Please click 'Initialize  AI for Assistance' first.")
    else:
        # Classify the user query
        query_type = classify_query(prompt1)

        # Store the user's question in the conversation history
        st.session_state.conversation_history.append({"user": prompt1, "type": query_type})

        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write("Response time:", time.process_time() - start)
        st.write(response['answer'])

        # Store the chatbot's response in the conversation history
        st.session_state.conversation_history.append({"bot": response['answer']})

        # Display the conversation history
        st.write("### Conversation History")
        for interaction in st.session_state.conversation_history:
            if "user" in interaction:
                st.write(f"**User:** {interaction['user']}")
            if "bot" in interaction:
                st.write(f"**Bot:** {interaction['bot']}")
        
        # With a streamlit expander
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
