# Import necessary modules and classes
import os
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.agent.introspective import IntrospectiveAgentWorker
from llama_index.llms.openai import OpenAI
from llama_index.agent.introspective import SelfReflectionAgentWorker

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"

# Define the user query
user_query = "The capital of France is London."

# Define the goal for the introspective worker agent
introspective_worker_agent_goal = """You are a fact checker, you check the user's input for inaccuracies and correct them."""

# Create a SelfReflectionAgentWorker with default settings
self_reflection_agent_worker = SelfReflectionAgentWorker.from_defaults(
        llm=OpenAI("gpt-4", system_prompt="", temperature=0.7),
        verbose=True,
    )

# Create an IntrospectiveAgentWorker with default settings
introspective_worker_agent = IntrospectiveAgentWorker.from_defaults(
        reflective_agent_worker=self_reflection_agent_worker,
        main_agent_worker=None,
        verbose=True,
    )

# Define the chat history
chat_history = [
    # System message setting the goal for the introspective worker agent
    ChatMessage(role=MessageRole.SYSTEM, content=introspective_worker_agent_goal)
]

# Convert the introspective worker agent into an agent
introspective_agent = introspective_worker_agent.as_agent(chat_history=chat_history)

# Process the user's query and get a response
response = introspective_agent.chat(user_query)

# Print the response
print(response)