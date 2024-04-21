from groq import Groq
import os
import json
import requests

client = Groq(api_key = os.getenv('GROQ_API_KEY'))
model = 'llama3-70b-8192'

def getStockPrice(stockName):
    """Get the current stock price for a given company game by querying the Flask API."""
    url = f'http://127.0.0.1:5000/price?stockName={stockName}'
    response = requests.get(url)
    if response.status_code == 200:
        return json.dumps(response.json())
    else:
        return json.dumps({"error": "API request failed", "status_code": response.status_code})

def run_conversation(user_prompt):
    # Send the conversation and available functions to the model
    messages=[
        {
            "role": "system",
            "content": "You are a function calling LLM that uses the data extracted from the getStockPrice function to answer questions around stock prices. Include the exchange, market cap and currency in the response."
        },
        {
            "role": "user",
            "content": user_prompt,
        }
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "getStockPrice",
                "description": "Get the price for a given stock.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "stockName": {
                            "type": "string",
                            "description": "The stock to get the price for. (e.g. 'Meta', 'Alphabet', 'Amazon', 'Apple', 'Netflix)",
                        }
                    },
                    "required": ["stockName"],
                },
            },
        }
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",  
        max_tokens=4096
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    # Check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function. Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "getStockPrice": getStockPrice,
        }  # we only use one function here, but you can have multiple functions
        messages.append(response_message)  # extend conversation with assistant's reply
        # Send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                stockName=function_args.get("stockName")
            )
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response
        second_response = client.chat.completions.create(
            model=model,
            messages=messages
        )  # get a new response from the model where it can see the function response
        return second_response.choices[0].message.content
    
user_prompt = "What is the stock price of Meta?"
print(run_conversation(user_prompt))