import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import pdf
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import ollama
import os

# Load, split, and retrieve documents from a local PDF file
def loadAndRetrieveDocuments(pdf_file_path: str) -> Chroma:
    loader = pdf.PyPDFLoader(pdf_file_path)
    documents = loader.load()
    textSplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documentSplits = textSplitter.split_documents(documents)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorStore = Chroma.from_documents(documents=documentSplits, embedding=embeddings)
    return vectorStore.as_retriever()

# Format a list of documents into a string
def formatDocuments(documents: list) -> str:
    return "\n\n".join(document.page_content for document in documents)

# Define the RAG chain function
def ragChain(pdf_file_path: str, question: str) -> str:
    retriever = loadAndRetrieveDocuments(pdf_file_path)
    retrievedDocuments = retriever.invoke(question)
    formattedContext = formatDocuments(retrievedDocuments)
    formattedPrompt = f"Question: {question}\n\nContext: {formattedContext}"
    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': formattedPrompt}])
    return response['message']['content']

# Gradio interface
interface = gr.Interface(
    fn=ragChain,
    inputs=["text", "text"],
    outputs="text",
    title="RAG Chain Question Answering",
    description="Enter a local PDF file path and a query to get answers from the RAG chain."
)

# Launch the app
interface.launch()