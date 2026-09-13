from datetime import datetime
import os
from dotenv import load_dotenv
import httpx
import streamlit as st
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

# Load environment variables from .env file
load_dotenv()

# Access environment variables
ollama_url = os.getenv("OLLAMA_BASE_URL")
openweather_key = os.getenv("OPENWEATHER_API_KEY")

# 1. Configure the Ollama model
model = OllamaModel("qwen2.5:3b", provider=OllamaProvider(base_url=ollama_url))

# 2. Initialize the Pydantic AI agent
agent = Agent(
    model,
    instructions=(
        "You are a helpful assistant equipped with tools to check the "
        "current date/time and fetch weather data. Use them when requested."
    ),
)


# 3. Define Tool 1: Current Date & Time
@agent.tool_plain
def get_current_time() -> str:
  """Get the current date and time."""
  return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 4. Define Tool 2: OpenWeather API integration
@agent.tool_plain
def get_weather(city: str) -> str:
  """Fetch the current weather for a given city using OpenWeatherMap.

  Args:
      city: The name of the city (e.g., 'London', 'Tokyo').
  """
  # Replace with your actual OpenWeatherMap API key
  api_key = openweather_key
  url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

  try:
    response = httpx.get(url, timeout=10.0)
    if response.status_code == 200:
      data = response.json()
      desc = data["weather"][0]["description"]
      temp = data["main"]["temp"]
      return f"The weather in {city} is {desc} with a temperature of {temp}°C."
    return f"Could not fetch weather for {city}. Check the spelling or API key."
  except Exception as e:
      return f"Error connecting to weather service: {str(e)}"


# 5. Streamlit Frontend Layout
st.set_page_config(page_title="Chat bot", page_icon="🤖")
st.title(" Assistant chat bot")
st.write(
    "This is a simple chat bot that can answer questions about the current time and weather. "
)

if "messages" not in st.session_state:
  st.session_state.messages = []

# Display prior chat messages
for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask me about the time or weather..."):
  st.session_state.messages.append({"role": "user", "content": prompt})
  with st.chat_message("user"):
    st.markdown(prompt)

  with st.chat_message("assistant"):
    with st.spinner("Thinking and checking tools..."):
      try:
        # Run the agent synchronously (Pydantic AI manages tool execution loops)
        result = agent.run_sync(prompt)
        response_text = result.output
      except Exception as e:
        response_text = f"An error occurred: {str(e)}"

      st.markdown(response_text)
      st.session_state.messages.append(
          {"role": "assistant", "content": response_text}
      )