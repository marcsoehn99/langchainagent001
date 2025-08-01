from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up the LLM
try:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        raise ValueError("OpenAI API key not properly configured")
    
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
        api_key=api_key
    )
    use_mock = False
    print("Using OpenAI LLM")
except Exception as e:
    print(f"OpenAI API key not found or invalid: {e}")
    print("Using mock responses.")
    use_mock = True

# Create a simple prompt template
prompt = PromptTemplate.from_template("What is the capital of {country}?")

# Create a chain: prompt -> llm -> string output
if use_mock:
    def mock_llm_response(input_text):
        return f"Mock response: The capital of {input_text.split()[-1].rstrip('?')} is [Mock Capital]"
    
    chain = prompt | mock_llm_response
else:
    chain = prompt | llm | StrOutputParser()

# Use the template with different inputs
countries = ["France", "Japan", "Brazil"]

print("=== PromptTemplate with LLM Example ===")
print()
print("Template:", prompt.template)
print()

for country in countries:
    result = chain.invoke({"country": country})
    print(f"Country: {country}")
    print(f"Response: {result}")
    print()

# Another example with multiple variables
greeting_prompt = PromptTemplate.from_template(
    "Hello {name}! How are you feeling today? It's {weather} outside. Please respond in a friendly way."
)

if use_mock:
    greeting_chain = greeting_prompt | mock_llm_response
else:
    greeting_chain = greeting_prompt | llm | StrOutputParser()

greeting_result = greeting_chain.invoke({"name": "Alice", "weather": "sunny"})
print("Multi-variable template:", greeting_prompt.template)
print("Response:", greeting_result)