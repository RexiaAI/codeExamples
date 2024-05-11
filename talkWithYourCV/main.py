import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import pdf
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage, MessageRole

LLM_MODEL = "llama3-gradient"
EMBED_MODEL = "nomic-embed-text"
PDF_FILE_PATH = "cv/my_cv.pdf"

class knowledge_base:
    def __init__(self):
        self.loader = pdf.PyPDFLoader(PDF_FILE_PATH)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        self.retriever = self.load_and_split_documents()
    
    def load_and_split_documents(self) -> Chroma:
        documents = self.loader.load()
        document_splits = self.text_splitter.split_documents(documents)
        vector_store = Chroma.from_documents(documents=document_splits, embedding=self.embeddings)
        return vector_store.as_retriever()

    def format_documents(self, documents: list) -> str:
        return "\n\n".join(document.page_content for document in documents)

class AIModelHandler:
    def __init__(self):
        self.llm = self.create_llm()

    def create_llm(self):
        return Ollama(model=LLM_MODEL, verbose=True, temperature=0.0)
    

    def chat(self, message, history):
        cv_knowledge_base = knowledge_base()
        retrieved_documents = cv_knowledge_base.retriever.invoke(message)
        formatted_context = cv_knowledge_base.format_documents(retrieved_documents)
        formatted_prompt = f"""You are the following C.V.: {formatted_context}
                            You answer questions about your content.
                            You write cover letters based on your content.
                            You are an expert in all things related to your content.
                            When provided with a job description, you produce a cover letter to apply for it.
                            \n\nQuestion: {message}\n\n"""

        response_stream = self.llm.stream_chat([ChatMessage(role=MessageRole.USER, content=formatted_prompt)])

        streaming_response = ""
        for response in response_stream:
            streaming_response = streaming_response + response.delta
            yield streaming_response

def main():
    ai_model_handler = AIModelHandler()

    gr.ChatInterface(ai_model_handler.chat).launch()    

if __name__ == "__main__":
    main()
