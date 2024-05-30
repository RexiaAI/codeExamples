"""Example using Mistral v0.3 with native function calling and OllamaFunctions from Langchain."""

import json
from typing import List
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_community.llms.ollama import Ollama
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.messages import BaseMessage
from langgraph.graph import MessageGraph, END

GOOGLE_API_KEY = "your_google_api_key_here"
SEARCH_ENGINE_ID = "your_search_engine_id_here"

class RexiaAIGoogleSearch(GoogleSearchAPIWrapper):
    """Workaround class to bind the function to the model. If anyone has a better way to do this, let me know."""
    def google_search(self, query: str) -> str:
        """Run query through GoogleSearch and parse result."""
        return super().run(query)

    def to_ollama_tool(self):
        """Return the tool as a JSON object for OllamaFunctions."""

        tool = [
            {
                "name": "google_search",
                "description": "Perform a Google search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query you wish to execute"
                            "e.g. 'How to make a cake'",
                        },
                    },
                    "required": ["query"],
                },
            }
        ]

        return tool

    def to_ollama_function_call(self):
        """Return the tool as a JSON object for OllamaFunctions."""
        function_call = {"name": "google_search"}

        return function_call

class LanggraphExample():
    """Example class for using langgraph."""
    def __init__(self):
        self.graph = MessageGraph()
        self.calling_agent = OllamaFunctions(model="mistral", format="json")
        self.result_agent = Ollama(model="mistral")
        self.tool = RexiaAIGoogleSearch(google_api_key=GOOGLE_API_KEY, google_cse_id=SEARCH_ENGINE_ID)
        self.calling_agent = self.calling_agent.bind_tools(tools=self.tool.to_ollama_tool(), function=self.tool.to_ollama_function_call())
        
    def create_graph(self):
        """Create a graph."""
        self.graph.add_node("Agent", self.calling_agent)
        self.graph.add_node("Agent With Tool Information", self.result_agent)
        self.graph.add_node("Call tools", self.call_tools)
        self.graph.add_conditional_edges("Agent", self.check_tools)
        self.graph.add_edge("Call tools", "Agent With Tool Information")
        self.graph.add_edge("Agent With Tool Information", END)
        self.graph.set_entry_point("Agent")
        
    def check_tools(self, state: List[BaseMessage]):
        """Check for tool calls."""
        tool_calls = state[-1].additional_kwargs.get("function_call", [])
        if len(tool_calls):
            print("We found tool calls: " + str(tool_calls))
            return "Call tools"
        else:
            return END
    
    def call_tools(self, state: List[BaseMessage]):
        """Call tools."""
        tool_call = state[-1].additional_kwargs.get("function_call", [])
        arguments = json.loads(tool_call["arguments"])
        query = arguments["query"]
        print("Calling tool with query: ", query)
        results = self.tool.google_search(query)
        print("Tool call results: ", results)
        return results
    
    def run(self, task: str):
        """Run the example."""
        runnable = self.graph.compile()
        graph_result = runnable.invoke(task)
        return graph_result
        
if __name__ == "__main__":
    example = LanggraphExample()
    example.create_graph()
    result = example.run("Search google and tell me the weather for today in New York City.")[-1]
    print(result)