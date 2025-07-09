from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, tool
from langchain_community.tools import TavilySearchResults
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

search_tool = TavilySearchResults()

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Get the current system time in the specified format.
    """
    return datetime.now().strftime(format)

tools = [search_tool, get_system_time]


agent = initialize_agent(tools=tools, llm=llm, agent="zero-shot-react-description", verbose=True)

agent.invoke("When was the last time bitcoin reached all time high? and how many days ago was that?")

