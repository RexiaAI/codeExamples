import re
from importlib.metadata import distributions
from typing import Tuple
from langchain_openai import ChatOpenAI

NIM_API_KEY = "your-nim-api-key-here"

llm = ChatOpenAI(
    model="meta/llama3-70b-instruct",
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NIM_API_KEY,
)

## Utility Functions

def get_installed_packages() -> str:
    """Get a list of installed packages."""
    return ", ".join(dist.metadata["Name"] for dist in distributions())

def strip_python_markdown(markdown: str) -> str:
    """Strip Python markdown syntax from a string."""
    return re.sub(r"```python\n(.*?)\n```", r"\1", markdown, flags=re.DOTALL)

## LLM Client

class LLMClient:
    def __init__(self, llm):
        self.llm = llm

    def invoke_llm(self, prompt: str) -> str:
        """Invoke the LLM with a prompt and return the response."""
        llm_response = self.llm.invoke(prompt)
        return llm_response.content

    def extract_function_and_usage(self, data: str) -> Tuple[str, str]:
        """Extract the function and usage from the data."""
        function_pattern = r"<function>\s*(.*?)\s*</function>"
        usage_pattern = r"<usage>\s*(.*?)\s*</usage>"

        function_match = re.search(function_pattern, data, re.DOTALL)
        usage_match = re.search(usage_pattern, data, re.DOTALL)

        function = function_match.group(1) if function_match else None
        usage = usage_match.group(1) if usage_match else None

        return function, usage

    def execute_from_response(self, input_data: str):
        """Execute the function and usage extracted from the response."""
        print("LLM Response:", input_data)

        function, usage = self.extract_function_and_usage(input_data)

        if function is None or usage is None:
            print("Error: Function or usage data could not be extracted.")
            return None

        function = strip_python_markdown(function)
        usage = strip_python_markdown(usage)

        if function and usage:
            print("Function:\n", function)
            print("Usage:\n", usage)

        global_scope = {}
        exec(function, global_scope)

        try:
            execution_result = eval(usage, global_scope)
            print("Execution result:", execution_result)
        except Exception as e:
            execution_result = str(e)
            print(
                f"An error occurred: {execution_result}\n\n Function: {function}\n\n Usage: {usage}"
            )

        return execution_result

def get_prompt(task: str) -> str:
    """Get the prompt for the task."""
    prompt = f"""
        ## Task Description
        As a code writing agent, your expertise lies in Python. You're part of a team working on a task, 
        and your role is to generate a Python function that aids in task completion.

        **Task:** {task}

        ## Output Format
        Your output should consist of two parts:
        1. A well-documented Python function that returns a single string. 
        The function should be task-specific and enclosed within `<function>` tags.
        Example: `<function>def add(a, b): return a + b</function>`

        2. A function call demonstrating the usage of the function, enclosed within `<usage>` tags.
        Example: `<usage>add(1, 2)</usage>`

        ## Requirements
        - The function should handle common edge cases and invalid inputs gracefully.
        - Use clear and descriptive variable names and comments to improve code readability.
        - Aim for efficient and optimized solutions, considering factors like time complexity and memory usage.
        - Do not use anything that requires an API key or any other form of authentication.
        - Do not use anything you do not have access to in your current environment.
        - Do not use print statements, your function should only return a value.
        - Print() statements will result in a failure of the task.
        - Your function should return the result, not print it.
        - Your usage should not print the result.

        ## Installed Packages
        You have access to the following installed packages and should only use these or built-in packages:
        {get_installed_packages()}
        
        """
    return prompt

def main():
    """Main function."""
    task = "What is the result of (17*3) + (12*5)?"

    prompt = get_prompt(task)

    llm_client = LLMClient(llm)
    response = llm_client.invoke_llm(prompt)

    result = str(llm_client.execute_from_response(response))

    answer = llm_client.invoke_llm(
        """
        Complete the following task using the information from function call result.
        Do not provide your own solution, use only the information from the function call result to solve the task.
        """
        + f"\n\ntask: {task}"
        + "\n\nFunction Call Result: "
        + f"\n\n{result}"
    )

    print(answer)

if __name__ == "__main__":
    main()