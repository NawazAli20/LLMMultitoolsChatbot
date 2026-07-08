import os

from dotenv import load_dotenv

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["YOU_API_KEY"] = os.getenv("YOU_API_KEY")

OPENWEATHER_API_KEY =os.getenv("OPENWEATHER_API_KEY")


#Get the llm model
from langchain.chat_models import init_chat_model

llm = init_chat_model(model="llama-3.3-70b-versatile", model_provider="Groq", temperature=0.8, max_tokens=1000)
#llm = init_chat_model(model="gpt-5.4-mini", model_provider="OpenAI", temperature=0.8, max_tokens=1000)
test_message = llm.invoke("hi").content
print(test_message)


import requests
# getWeather tool 
from langchain.tools import tool

url = "https://api.openweathermap.org/data/2.5/weather?"

@tool
def getWeather(location:str,zipcode:str)->str:
    """Returns weather information based on the city and/or zipcode"""
    params={
        "zip":zipcode,
        "q":location,
        "appid":OPENWEATHER_API_KEY,
        "units":"imperial" #imperial
    }
    try:
        response = requests.get(url,params=params,timeout=10)
        return response.json()
    except:
        return f"Does not have the weather information"

## Add Tavily Search tool 

from langchain_tavily import TavilySearch 

web_search = TavilySearch(
    max_results=3,
    topic="news",
    seatch_depth="basic"
)

## Add duckduckgosearchrun 

from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults

duck_search = DuckDuckGoSearchResults(); 

# Add youdotcom search 
from langchain_youdotcom import YouSearchTool

you_search = YouSearchTool()

## Message formatting 
from langchain.messages import AIMessage, HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You are an helpful assistant. " \
    "For normal conversation, answer directly without using tools." \
    "for weather related query use getWeather tool" \
    "for for news and current affairs " \
    "use web_search tool, otherwise answer using the llm model without using the tool")
]

user_input = input("What is your query: ")
messages.append(HumanMessage(content=user_input))

## bind the tool with the model and append the AI message 

llm_with_tools = llm.bind_tools([getWeather,web_search,duck_search,you_search])

aiMessage = llm_with_tools.invoke(messages)
#print("AIMessage:",aiMessage)
messages.append(aiMessage)

# Append tool call message
tool_name = ""
for tool_call in aiMessage.tool_calls:
    if(tool_call["name"]==getWeather.name):
        tool_call_message = getWeather.invoke(tool_call)
        tool_name = tool_call["name"]
    elif(tool_call["name"]==web_search.name):
        tool_call_message = web_search.invoke(tool_call)
        tool_name = tool_call["name"]
    elif(tool_call["name"]==duck_search.name):
        tool_call_message = duck_search.invoke(tool_call)
        tool_name = tool_call["name"]
    elif(tool_call["name"]==you_search.name):
        tool_call_message = you_search.invoke(tool_call)
        tool_name = tool_call["name"]
    
    #print(tool_call_message.name)
    messages.append(tool_call_message)

## Respons with tool calls 
final_response = llm_with_tools.invoke(messages)
print("\n..................")
print("Tool name: ", tool_name)
print(final_response.content)
print("\n..................")

##Response with structured output 

# from pydantic import BaseModel, Field 

# class FormattedOutput(BaseModel):
#     day: str 
#     low_temperarure:float
#     high_tempature:float
#     wind:str

# class FiveDaysForcast():
#     location:str
#     forcast:list[FormattedOutput]

# llm_with_structured_op = llm.with_structured_output(FormattedOutput)
# #llm_with_structured_op = llm.with_structured_output(FiveDaysForcast)

# final_reponse = llm_with_structured_op.invoke(messages)
# print("...............")
# print(final_reponse)
# print("...............")
