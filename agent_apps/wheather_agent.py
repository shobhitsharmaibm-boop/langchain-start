import os
import requests
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from dotenv import load_dotenv
from application_types.weather import GeocodeResponse, Weather

load_dotenv()


API_KEY = os.getenv("WEATHER_API_KEY")


def get_latitude_longitude(location: str) -> tuple[float, float]:
    """Returns the latitude and longitude for a given location."""
    # In a real scenario, we would call a geocoding API here.
    # We return mock data for demonstration.

    response: GeocodeResponse = requests.get(f"https://maps.googleapis.com/maps/api/geocode/json?address={location}&key={API_KEY}")
    if response.status_code == 200:
        data = response.json()
        if data.get("status") == "OK":
            geometry = data["results"][0]["geometry"]["location"]
            return (geometry["lat"], geometry["lng"])
    return None

def current_weather(latitude: float, longitude: float) -> str:
    """Returns the weather for a given latitude and longitude."""
    # In a real scenario, we would call a weather API here.
    # We return mock data for demonstration.
    response: Weather = requests.get(f"https://weather.googleapis.com/v1/currentConditions:lookup?key={API_KEY}&location.latitude={latitude}&location.longitude={longitude}")
    if response.status_code == 200:
        data = response.json()
        if data.get("status") == "OK":
            return data
    return None

@tool
def get_weather(location: str) -> str:
    """Returns the weather for a given location."""
    # In a real scenario, we would call a weather API here.
    # We return mock data for demonstration.
    lat_lng = get_latitude_longitude(location)
    if lat_lng:
        return current_weather(lat_lng[0], lat_lng[1])
    return "Location not found."

def start_weather_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    tools = [get_weather, get_latitude_longitude]
    
    # Define the system prompt
    system_prompt = 'You are a helpful weather assistant. Use the provided tools to answer the user\'s question about the weather.'
    
    # Construct the agent
    agent = create_agent(
        model=llm,
        tools=tools,    
        system_prompt=system_prompt
    )
    
    # Test the agent with a weather-related query
    user_query = "What is the weather like in jalandhar location?"
    print(f"User: {user_query}")
    
    response = agent.invoke({"messages": [("user", user_query)]})
    print(f"Agent: {response['messages'][-1].content}")

if __name__ == "__main__":
    start_weather_agent()
