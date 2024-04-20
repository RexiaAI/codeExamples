from crewai import Agent, Task, Crew, Process
import os

os.environ["OPENAI_API_BASE"] = 'https://api.groq.com/openai/v1'
os.environ["OPENAI_MODEL_NAME"] ='llama3-70b-8192'
os.environ["OPENAI_API_KEY"] ='your API key here'

codeSnippet = 'print("Hello, World!")'

explainer = Agent(role="code explainer", 
                        goal="Take in a code snippet and explain how it works and what it does in natural language",
                        backstory="You are a Senior Software engineer at a tech company and are tasked with explaining a code snippet to a blog writer who is writing a blog post on the codebase.",
                        verbose=True,
                        allow_delegation=False)

blogWriter = Agent(role="blog writer",
                        goal="Take an explanation of a code snippet provided by the code explainer agemt and write an explanatory blog post on it",
                        backstory="You are a blog writer who is writing a blog post on a codebase. You need to understand the code explanation provided by the code exxplainer agent and write a blog post on it.",
                        verbose=True,
                        allow_delegation=False)

explainSnippet = Task(description=f"Explain '{codeSnippet}' a code snippet",
                      agent=explainer,
                      expected_output="An explanation of the code snippet in natural language")

writeBlogPost = Task(description=f"Write a blog post on this code snippet: '{codeSnippet}', using the explanation provided by the code explainer agent. Include relevant sections of the code with your explanations.",
                        agent=blogWriter,
                        expected_output="A blog post containing code an explanations of the code. The explanation should be in natural language and should be based on the explanation provided by the previous agent for the code")

crew = Crew(agents=[explainer, blogWriter], 
            tasks=[explainSnippet, writeBlogPost],
            verbose=2,
            process=Process.sequential)
                      
output = crew.kickoff()
print(output)