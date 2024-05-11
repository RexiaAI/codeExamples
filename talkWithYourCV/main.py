import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import pdf
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage, MessageRole

# Constants for model and embedding identifiers
LLM_MODEL = "llama3-gradient"
EMBED_MODEL = "nomic-embed-text"
PDF_FILE_PATH = "cv/my_cv.pdf"

class knowledge_base:
    def __init__(self):
        # Initialize the PDF loader with the path to the PDF file
        self.loader = pdf.PyPDFLoader(PDF_FILE_PATH)
        # Initialize the text splitter with specific chunk size and overlap
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        # Load the embeddings model
        self.embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        # Load, split, and retrieve documents from the PDF
        self.retriever = self.load_and_split_documents()
    
    def load_and_split_documents(self) -> Chroma:
        # Load documents from the PDF
        documents = self.loader.load()
        # Split documents into smaller chunks
        document_splits = self.text_splitter.split_documents(documents)
        # Create a vector store from the document splits and embeddings
        vector_store = Chroma.from_documents(documents=document_splits, embedding=self.embeddings)
        # Return the vector store as a retriever
        return vector_store.as_retriever()

    def format_documents(self, documents: list) -> str:
        # Format the documents into a single string separated by new lines
        return "\n\n".join(document.page_content for document in documents)

class AIModelHandler:
    def __init__(self):
        # Create an instance of the LLM
        self.llm = self.create_llm()
        # Create an instance of the knowledge base
        self.cv_knowledge_base = knowledge_base()
        # Retrieve documents using the retriever
        self.retrieved_documents = self.cv_knowledge_base.retriever.invoke(None)
        # Format the retrieved documents for display
        self.formatted_context = self.cv_knowledge_base.format_documents(self.retrieved_documents)

    def create_llm(self):
        # Initialize the LLM with specific model settings
        return Ollama(model=LLM_MODEL, verbose=True, temperature=0.0)
    
    def format_prompt(self, context, message):
        # Format the prompt for the LLM
        return f"""You are the following C.V.: {context}\n\nQuestion: {message}\n\n"""
    
    def chat(self, message, history):
        # Generate a chat response using the LLM
        formatted_prompt = self.format_prompt(self.formatted_context, message)
        response_stream = self.llm.stream_chat([ChatMessage(role=MessageRole.USER, content=formatted_prompt)])

        streaming_response = ""
        for response in response_stream:
            streaming_response += response.delta
            yield streaming_response

def main():
    # Create an instance of the AI model handler
    ai_model_handler = AIModelHandler()
    # Launch a Gradio chat interface with the AI model handler
    gr.ChatInterface(ai_model_handler.chat).launch()    

if __name__ == "__main__":
    main()