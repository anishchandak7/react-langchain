import os
import warnings
from typing import List, Union

from dotenv import load_dotenv
from langchain.agents.format_scratchpad.log import format_log_to_str
from langchain.agents.output_parsers.react_single_input import \
    ReActSingleInputOutputParser
from langchain.prompts.prompt import PromptTemplate
from langchain.tools import Tool, tool
from langchain.tools.render import render_text_description
from langchain_core.agents import AgentAction, AgentFinish
from langchain_openai import ChatOpenAI
from callbacks import AgentCallbackHandler
warnings.filterwarnings('ignore')


load_dotenv()

@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"get_text_length enter with {text=}")
    # stripping away non alphabetic characters just in case.
    text = text.strip("'\n").strip('"')
    return len(text)

def find_tool_by_name(tools: List[Tool], tool_name: str) -> str:
    """
    The function `find_tool_by_name` takes a list of Tool objects and a tool name as input, and returns
    the Tool object with the matching name or raises a ValueError if not found.
    
    :param tools: List of Tool objects, representing different tools available
    :type tools: List[Tool]
    :param tool_name: The `tool_name` parameter is a string that represents the name of the tool you are
    searching for in the list of tools
    :type tool_name: str
    :return: The function `find_tool_by_name` is returning the tool object that matches the given
    `tool_name`. If no tool with the specified name is found in the list of tools, it raises a
    `ValueError` with a message indicating that the tool with the given name was not found.
    """
    for t in tools:
        if t.name == tool_name:
            return t
    raise ValueError(f"Tool with name {tool_name} not found")

if __name__ == "__main__":

    tools = [get_text_length]
    
    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
    """
    prompt = PromptTemplate.from_template(template=template).partial(tools=render_text_description(tools), tool_names=", ".join([t.name for t in tools]))   
    llm = ChatOpenAI(
        temperature=0,
        stop=["\nObservation", "Observation"],
        api_key=os.environ.get("OPENAI_API_KEY"),
        callbacks=[AgentCallbackHandler()]
    )
    intermediate_steps = [] # To keep track of history.
    agent=(
        {
            "input": lambda x:x['input'],
            "agent_scratchpad":lambda x:format_log_to_str(x['agent_scratchpad'])
        }
        |prompt
        |llm
        |ReActSingleInputOutputParser()
    )
    agent_step = ""
    while not isinstance(agent_step, AgentFinish):
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {"input": "What is the text length of ANISH in characters?", "agent_scratchpad": intermediate_steps}
        )
        print(agent_step)
        
        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tools=tools, tool_name=tool_name)
            tool_input = agent_step.tool_input
            
            observation = tool_to_use.func(str(tool_input))
            print(f"{observation=}")
            intermediate_steps.append((agent_step, str(observation)))
        
    if isinstance(agent_step, AgentFinish):
        print(agent_step.return_values)