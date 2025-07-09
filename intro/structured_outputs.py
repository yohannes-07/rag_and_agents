from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# class Country(BaseModel):
#     """"Information about a country."""
#     name: str = Field(description="The name of the country.")
#     capital: str = Field(description="The capital city of the country.")
#     population: int = Field(description="The population of the country.")
#     area: float = Field(description="The area of the country in square kilometers.")


# structured_llm = llm.with_structured_output(Country)
# response = structured_llm.invoke("Tell me about Ethiopia")

# print(response)


# from typing_extensions import Annotated, TypedDict
# from typing import Optional

# class Joke(TypedDict):

#     """Joke to tell user."""

#     setup: Annotated[str, ..., "The setup of the joke"]

#     # Alternatively, we could have specified setup as:

#     # setup: str                    # no default, no description
#     # setup: Annotated[str, ...]    # no default, no description
#     # setup: Annotated[str, "foo"]  # default, no description

#     punchline: Annotated[str, ..., "The punchline of the joke"]
#     rating: Annotated[Optional[int], None, "How funny the joke is, from 1 to 10"]


# structured_llm = llm.with_structured_output(Joke)

# response = structured_llm.invoke("Tell me a joke about cats")
# print(response)

json_schema = {
    "title": "joke",
    "description": "Joke to tell user.",
    "type": "object",
    "properties": {
        "setup": {
            "type": "string",
            "description": "The setup of the joke",
        },
        "punchline": {
            "type": "string",
            "description": "The punchline to the joke",
        },
        "rating": {
            "type": "integer",
            "description": "How funny the joke is, from 1 to 10",
            "default": None,
        },
    },
    "required": ["setup", "punchline"],
}
structured_llm = llm.with_structured_output(json_schema)

response = structured_llm.invoke("Tell me a joke about cats")
print(response)