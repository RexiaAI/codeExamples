from crewai import Crew, Process, Agent, Task
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI

# set up the job
prompt = "A game of snake"

# The model that does the coding. Temperature to 0.5 to allow more creativity in this task
coder = Ollama(model="wizardlm2:7b", temperature=0.5)

# The model that does the project management, temperature to 0.5 to allow more creativity in this task
project_manager = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, api_key="your-openai-api-key")

# The model that manages the crew, temperature to 0 so the manager does not try to be creative.
crew_manager = ChatOpenAI(model="gpt-4", temperature=0, api_key="your-openai-api-key") 

# create the agents
project_manager = Agent(
    role='Project Manager',
    backstory='''You are a Senior Project Manager. You are responsible for deciding how the project should be structured.
                 You break the project down into tasks that can be performed by developers.''',
    goal="A list of tasks required to complete the overall project.",
    verbose=True,
    llm=project_manager
)
    
coder = Agent(
    role='Python coder',
    backstory='''You are a Senior Developer. You are responsible for writing high-quality Python code.
                 You write simple, readable Python code. You are not responsible for reviewing the code.
                 Your ultimate goal is to produce high-quality, maintainable Python code.
                 You perform only one iteration at a time.''',
    goal="Well written and structured python code that is simple and efficient",
    verbose=True,
    llm=coder
)

# Create a task for project management
task_project_nagement = Task(
    description=f"""Decide how the project,'{prompt}',should be structured. Break down the project into tasks that can be performed by developers. """,
    agent=project_manager,
    expected_output="A list of tasks required to complete the overall project."
)
                 
# Create a task for coding
task_code = Task(
    description="""Write python code as instructed by the manager. """,
    agent=coder,
    expected_output="Well written and structured code that is simple and efficient"
)
   
# Establishing the crew with a hierarchical process
project_crew = Crew(
    tasks=[task_project_nagement, task_code],  # Tasks to be delegated and executed under the manager's supervision
    agents=[project_manager, coder], # Agents that will be delegated to the crew
    manager_llm=crew_manager, # LLM to manage the agents, required for hierarchical process
    process=Process.hierarchical  # Specifies the hierarchical management approach
)
result = project_crew.kickoff()

# Print the result
print(result)