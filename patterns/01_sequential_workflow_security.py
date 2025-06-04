from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from dotenv import load_dotenv
from utils import SequentialCodebase, CodebaseGenerator
import time
from copy import deepcopy

load_dotenv()


class CodeReviewState(TypedDict):
    input: str
    code: str
    review: str
    refactored_code: str
    unit_tests: str
    codebase: CodebaseGenerator


llm = ChatOpenAI(model="gpt-4.1-nano")

coder_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Junior Software Engineer. Write awful, insecure code based on requirements."),
    ("human", "{input}")
])

reviewer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Senior Software Engineer Code Reviewer. Provide constructive feedback focusing on readability, efficiency, and best practices."),
    ("human", "Review this code:\n{code}")
])

refactorer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Senior Software Engineer Refactoring Expert. Implement the suggested improvements focusing on security."),
    ("human",
     "Original code:\n{code}\n\nReview feedback:\n{review}\n\nRefactor accordingly:")
])

tester_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Test Expert. Create a set of tests to ensure any future changes to the original code wont break the expected functionality. Make sure to add the original code in the tests"),
    ("human",
     "Original code:\n{code}\n")
])


def save_state(agent_name, args):
    codebase = args.get("codebase")
    copy = deepcopy(args)
    del copy["codebase"]

    codebase.create_folder()
    codebase.write_json_file(f"{agent_name}.json", copy)

def time_node_execution(fn):
    agent_name = fn.__name__.split("_agent")[0]

    def wrapper(*args, **kwargs):
        print(f"🔄 Starting {agent_name}...")
        start = time.time()
        
        save_state(agent_name, args[0])

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
    response = llm.invoke(tester_prompt.format_messages(
        code=state["refactored_code"]))
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
    task = "Write a basic Flask API"

    print("Running sequential workflow...")
    codebase = SequentialCodebase("01_sequential_workflow_security", task)

    result = workflow.invoke({"input": task, "codebase": codebase})

    codebase.generate(result)

    print("=== WORKFLOW COMPLETED ===")
