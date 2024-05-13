from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.node_parser import HTMLNodeParser

LLM_MODEL = "dolphin-mistral"
EMBED_MODEL = "nomic-embed-text"
URL = "https://www.llamaindex.ai/"

class WebDriverManager:
    """Manages the lifecycle and operations of the Selenium WebDriver."""
    
    def __init__(self):
        """Initializes the WebDriver with necessary options."""
        self.options = Options()
        self.driver = self._create_driver()

    def _create_driver(self) -> webdriver.Chrome:
        """Private method to create a Chrome WebDriver instance."""
        try:
            return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=self.options)
        except Exception as e:
            print(f"Failed to create WebDriver: {e}")
            raise

    def navigate_to(self, url: str) -> None:
        """Navigates to a specified URL using the WebDriver."""
        try:
            self.driver.get(url)
        except Exception as e:
            print(f"Failed to navigate to {url}: {e}")
            raise

    def close(self) -> None:
        """Closes the WebDriver, effectively quitting the browser session."""
        self.driver.quit()

class LlamaIndexClient:
    """Client to handle operations related to LlamaIndex models and data processing."""
    
    def __init__(self):
        """Initializes the LlamaIndex models."""
        self.embeddings = OllamaEmbedding(model_name=EMBED_MODEL)
        self.llm = Ollama(model=LLM_MODEL, verbose=True, temperature=0.0, json_mode=True)

    def create_query_engine(self, url: str, html: str) -> VectorStoreIndex:
        """Creates a query engine from HTML content for analysis."""
        document = Document(id_=url, text=html)
        parser = HTMLNodeParser()
        nodes = parser.get_nodes_from_documents([document])
        vector_index = VectorStoreIndex(nodes=nodes, embed_model=self.embeddings)
        return vector_index.as_query_engine(llm=self.llm)

class AIAssistant:
    """AI Assistant for orchestrating web scraping and data analysis tasks."""
    
    def __init__(self):
        """Initializes the components used by the AI Assistant."""
        self.web_driver_manager = WebDriverManager()
        self.llama_index_client = LlamaIndexClient()

    def process_url(self, url: str):
        """Processes a URL to extract and analyze its content."""
        try:
            self.web_driver_manager.navigate_to(url)
            page_source = self.web_driver_manager.driver.page_source
            current_url = self.web_driver_manager.driver.current_url
            query_engine = self.llama_index_client.create_query_engine(current_url, page_source)
            return query_engine.query(self._get_analysis_prompt())
        finally:
            self.web_driver_manager.close()

    def _get_analysis_prompt(self) -> str:
        """Returns the analysis prompt for querying the HTML content."""
        return """
            Analyze and extract all human-readable text, 
                identify the most important keywords based on frequency and relevance, 
                and list all hyperlinks (both internal and external). 
                Provide the results in a structured report format.
                Produce only this output.

                Instructions:
                1. Parse the HTML to extract all human-readable text. This includes text within paragraph tags, headings, lists, and other text-bearing elements. Exclude any scripts, CSS, inline styles, and irrelevant metadata.
                2. Analyze the extracted text to determine the most important keywords. Consider the frequency of words and their semantic importance in the context.
                3. Extract all links (anchor tags) and provide their href values.
                4. Format your report in JSON as follows:

                {
                "Main Content": "<Extracted human-readable text>",
                "Most Important Keywords": ["List of keywords"],
                "Links": ["List of all href values"]
                }
            """

def main():
    """Main function to execute the AI Assistant's tasks."""
    assistant = AIAssistant()
    result = assistant.process_url(URL)
    print(result)

if __name__ == "__main__":
    main()