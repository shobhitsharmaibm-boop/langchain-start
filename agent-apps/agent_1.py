from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

def start_agent_one():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    tools = [get_word_length]
    
    # Get the prompt to use - you can modify this!
    system_prompt = 'You are a helpful assistant. Use the tools to answer the user\'s question.'
    
    # Construct the OpenAI Tools agent
    agent_one = create_agent(
        model=llm,
        tools=tools,    
        system_prompt=system_prompt
    )
    
    response = agent_one.invoke({"messages": [("user", "How many letters are in the word 'supercalifragilisticexpialidocious'?")]})
    print(response['messages'][-1].content)


def start_agent_two():
    llm = ChatOllama(model="llama3:latest", temperature=0)
    # llama3:latest lacks native tool support in Ollama, so we pass an empty tools list 
    # to bypass the bind_tools error and let the LLM answer directly!
    tools = []
    
    # Get the prompt to use - you can modify this!
    system_prompt = 'You are a helpful assistant.'
    
    # Construct the base agent (which skips bind_tools if no tools are provided)
    agent_two = create_agent(
        model=llm,
        tools=tools,    
        system_prompt=system_prompt
    )
    
    response = agent_two.invoke({"messages": [("user", "How many letters are in the word 'supercalifragilisticexpialidocious'?")]})
    print(response['messages'][-1].content)

if __name__ == "__main__":
    start_agent_two()
    
