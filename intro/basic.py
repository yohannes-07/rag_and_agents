from typing import Sequence, List
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import MessageGraph, END
from intro.chains import generation_chain, reflection_chain


graph = MessageGraph()

GENERATE = 'generate'
REFLECT = 'reflect'

def generate_node(state):
    return generation_chain.invoke(
        {
             "messages": state
        }
    )

def reflect_node(state):
    response =  reflection_chain.invoke(
        {
            "messages": state
        }
    )

    return [HumanMessage(content=response.content)]

graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)

graph.set_entry_point(GENERATE)

def should_continue(state):
    if len(state) > 4:
        return END
    
    return REFLECT


graph.add_conditional_edges(GENERATE, should_continue, [REFLECT, END])
graph.add_edge(REFLECT, GENERATE)

app = graph.compile()

print(app.get_graph().draw_mermaid())
app.get_graph().print_ascii()

response = app.invoke(HumanMessage(content="AI agents taking over content creation"))
print(response)