# Import necessary libraries
from crewai import Agent, Task, Crew, Process
import os

# Setting environment variables for the API (this should generally be done in a separate .env file)
os.environ["OPENAI_API_BASE"] = 'https://api.groq.com/openai/v1' #Groq API base_url
os.environ["OPENAI_MODEL_NAME"] = 'llama3-70b-8192' #Model you wish to use, see https://console.groq.com/docs/models
os.environ["OPENAI_API_KEY"] = 'your API key here' #Your Groq API key

# Code snippet to be explained
codeSnippet = 'print("Hello, World!")'

# Define the explainer agent
explainer = Agent(
    role="code explainer",
    goal="Take in a code snippet and explain how it works and what it does in natural language",
    backstory=("You are a Senior Software engineer at a tech company and are tasked with explaining a code snippet to a blog writer who is writing a blog post on the code base."),
    verbose=True,
    allow_delegation=False
)

# Define the blog writer agent
blogWriter = Agent(
    role="blog writer",
    goal="Take an explanation of a code snippet provided by the code explainer agent and write an explanatory blog post on it",
    backstory=("You are a blog writer who is writing a blog post on a codebase. You need to understand the code explanation provided by the code explainer agent and write a blog post on it."),
    verbose=True,
    allow_delegation=False
)

# Task to explain the code snippet
explainSnippet = Task(
    description=f"Explain '{codeSnippet}' a code snippet",
    agent=explainer,
    expected_output="An explanation of the code snippet in natural language"
)

# Task to write a blog post based on the explanation
writeBlogPost = Task(
    description=f"Write a blog post on this code snippet: '{codeSnippet}', using the explanation provided by the code explainer agent. Include relevant sections of the code with your explanations.",
    agent=blogWriter,
    expected_output="A blog post containing code and explanations of the code. The explanation should be in natural language and should be based on the explanation provided by the previous agent for the code"
)

# Define the crew and process
crew = Crew(
    agents=[explainer, blogWriter],
    tasks=[explainSnippet, writeBlogPost],
    verbose=2,
    process=Process.sequential
)

# Execute the process and print the output
output = crew.kickoff()
print(output)