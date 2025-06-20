from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
from dotenv import load_dotenv
from utils import SequentialCodebase
from pydantic import BaseModel, Field

load_dotenv()


class CodeReviewState(TypedDict):
    input: str
    code: str
    review: str
    refactored_code: str
    unit_tests: str


llm = ChatOpenAI(model="gpt-4.1-nano")

coder_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Senior Software Engineer. Write clean, well-structured Python code based on requirements."),
    ("human", "{input}")
])

reviewer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Code Reviewer. Provide constructive feedback focusing on readability, efficiency, and best practices."),
    ("human", "Review this code:\n{code}")
])

refactorer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Refactoring Expert. Implement the suggested improvements while maintaining functionality."),
    ("human",
     "Original code:\n{code}\n\nReview feedback:\n{review}\n\nRefactor accordingly:")
])

tester_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Test Expert. Create a set of tests to ensure any future changes to the original code wont break the expected functionality. Make sure to add the original code functions in the tests, for the time being. Use the additional feedback if provided"),
    ("human",
     "Original code:\n{code}\n\nFeedback: {feedback}")
])


import time

def time_node_execution(fn):
    agent_name = fn.__name__.split("_agent")[0]

    def wrapper(*args, **kwargs):
        print(f"🔄 Starting {agent_name}...")
        start = time.time()
        result = fn(*args, **kwargs)
        end = time.time()
        print(f"✅ {agent_name} completed in {end - start:.2f}s")
        return result

    return wrapper


@time_node_execution
def coder_agent(state: CodeReviewState) -> CodeReviewState:
    response = llm.invoke(coder_prompt.format_messages(input=state["input"]))
    return {"code": response.content}

@time_node_execution
def reviewer_agent(state: CodeReviewState) -> CodeReviewState:
    response = llm.invoke(reviewer_prompt.format_messages(code=state["code"]))
    return {"review": response.content}

@time_node_execution
def refactorer_agent(state: CodeReviewState) -> CodeReviewState:
    response = llm.invoke(refactorer_prompt.format_messages(
        code=state["code"], review=state["review"]))
    return {"refactored_code": response.content}

@time_node_execution
def tester_agent(state: CodeReviewState) -> CodeReviewState:
    feedback = state.get("feedback", "")
    response = llm.invoke(tester_prompt.format_messages(
        code=state["refactored_code"], feedback=feedback))
    return {"unit_tests": response.content}


builder = StateGraph(CodeReviewState)
builder.add_node("coder", coder_agent)
builder.add_node("reviewer", reviewer_agent)
builder.add_node("refactorer", refactorer_agent)
builder.add_node("tester", tester_agent)

builder.add_edge(START, "coder")
builder.add_edge("coder", "reviewer")
builder.add_edge("reviewer", "refactorer")
builder.add_edge("refactorer", "tester")
builder.add_edge("tester", END)

workflow = builder.compile()

if __name__ == "__main__":
    task = "Write a function that validates email addresses using regex"

    print("Running sequential workflow...")
    result = workflow.invoke({"input": task})

    codebase = SequentialCodebase("01_sequential_workflow", task)
    codebase.generate(result)

    print("=== WORKFLOW COMPLETED ===")
