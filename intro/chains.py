from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

load_dotenv()

generation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a twitter techy influencer assistant tasked with writing excellent tweets"
         "Generate the best tweet possible for the user's input"
         "If the user responds with critique, improve the tweet based on the feedback",
         ),
         MessagesPlaceholder(variable_name="messages")
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a viral twitter influence grading tweets. Generate critque and recommendations for the user's request"
         "Provide a detailed critique of the tweet and suggest improvements such as length, tone, and engagement",
         ),
         MessagesPlaceholder(variable_name="messages")
    ]
)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm

