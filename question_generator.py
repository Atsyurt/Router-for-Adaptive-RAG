import os
import json
import argparse
import time
from typing import TypedDict

import requests
from bs4 import BeautifulSoup
from langchain_core.language_models import BaseChatModel
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
import wikipedia

# --- 1. Define the State for our Graph ---
class GraphState(TypedDict):
    topic: str
    url: str
    document: str
    question: str
    answer: str
    source_chunk: str
    llm: BaseChatModel # <-- LLM is part of the state
    num_generated: int
    max_generations: int
    all_results: list
    used_topics: list[str]

# --- 2. Define the Nodes for our Graph ---

def generate_topic(state):
    print("\n---NODE: GENERATING TOPIC---")
    llm = state['llm'] # <-- Get LLM from state
    used_topics = state.get('used_topics', [])

    class Topic(BaseModel):
        topic: str = Field(description="A specific and interesting topic, suitable for a Wikipedia search.")

    prompt_text = "Generate a single, random, and interesting topic."
    if used_topics:
        prompt_text += " The topic should NOT be about any of the following: {excluded_topics}."

    prompt = PromptTemplate.from_template(prompt_text)
    topic_gen_chain = prompt | llm.with_structured_output(Topic)
    generated_topic = topic_gen_chain.invoke({"excluded_topics": ", ".join(used_topics)})

    print(f"Generated Topic: {generated_topic.topic}")
    return {"topic": generated_topic.topic}

def search_wikipedia(state):
    
    print("---NODE: SEARCHING WIKIPEDIA---")
    topic = state['topic']
    try:
        page_title = wikipedia.search(topic, results=1)[0]
        page = wikipedia.page(page_title, auto_suggest=False)
        url = page.url
        print(f"Found URL: {url}")
        return {"url": url}
    except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError, IndexError) as e:
        print(f"Error searching Wikipedia for '{topic}': {e}")
        return {"url": None}

def fetch_content(state):
    print("---NODE: FETCHING CONTENT---")
    url = state['url']
    if not url:
        return {"document": None}
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        print("Successfully fetched and cleaned document.")
        return {"document": text}
    except requests.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return {"document": None}

def generate_question(state):
    print("---NODE: GENERATING QUESTION---")
    llm = state['llm'] # <-- Get LLM from state
    document = state['document']
    if not document:
        return {"question": None, "source_chunk": None}
    source_chunk = document[:4000]
    prompt = PromptTemplate.from_template(
        """Based *only* on the context provided below, generate a single, clear, and concise question that can be answered directly from this text.
        The question should be simple and factual (Type A).

        CONTEXT:
        ---
        {context}
        ---

        QUESTION:"""
    )
    question_chain = prompt | llm
    generated_question = question_chain.invoke({"context": source_chunk})
    print(f"Generated Question: {generated_question.content}")
    return {"question": generated_question.content, "source_chunk": source_chunk}

def generate_answer(state):
    print("---NODE: GENERATING ANSWER---")
    llm = state['llm'] # <-- Get LLM from state
    question = state['question']
    source_chunk = state['source_chunk']
    if not question:
        return {"answer": None}
    
    prompt = PromptTemplate.from_template(
        """Based *only* on the context provided below, provide a concise and direct answer to the following question.

        CONTEXT:
        ---
        {context}
        ---

        QUESTION: {question}

        ANSWER:"""
    )
    answer_chain = prompt | llm
    generated_answer = answer_chain.invoke({"context": source_chunk, "question": question})
    print(f"Generated Answer: {generated_answer.content}")
    return {"answer": generated_answer.content}

def store_result(state):
    """NEW NODE: Stores the result of the current run and increments the counter."""
    print("---NODE: STORING RESULT---")
    topic = state["topic"]
    result = {
        "topic": topic,
        "url": state["url"],
        "question": state["question"],
        "answer": state["answer"],
        "question_type": "Type A",
        "source_chunk": state["source_chunk"]
    }
    
    # Idiomatic state update: return only the changed keys
    updated_results = state["all_results"] + [result]
    updated_count = state["num_generated"] + 1
    updated_used_topics = state["used_topics"] + [topic] # <-- ADD THIS LINE
    
    return {
        "all_results": updated_results, 
        "num_generated": updated_count,
        "used_topics": updated_used_topics # <-- ADD THIS LINE
    }

# --- 3. Define Conditional Edges ---

def should_continue(state):
    """Conditional edge: decides whether to continue the loop or end."""
    num_generated = state.get("num_generated", 0)
    max_generations = state.get("max_generations", 1)
    
    print("\n---EDGE: DECIDING TO CONTINUE?---")
    print(f"  Generated so far: {num_generated}")
    print(f"  Max generations requested: {max_generations}")
    
    if num_generated >= max_generations:
        print("  Decision: END graph.")
        return "end"
    else:
        # ADD DELAY TO AVOID HITTING API QUOTAS
        delay = 2 # seconds
        print(f"  Decision: CONTINUE to next loop after a {delay}-second delay.")
        time.sleep(delay)
        return "continue"

# --- 4. Build the Graph ---

def build_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("generate_topic", generate_topic)
    workflow.add_node("search_wikipedia", search_wikipedia)
    workflow.add_node("fetch_content", fetch_content)
    workflow.add_node("generate_question", generate_question)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("store_result", store_result)

    workflow.set_entry_point("generate_topic")
    workflow.add_edge("generate_topic", "search_wikipedia")
    workflow.add_edge("search_wikipedia", "fetch_content")
    workflow.add_edge("fetch_content", "generate_question")
    workflow.add_edge("generate_question", "generate_answer")
    workflow.add_edge("generate_answer", "store_result")
    
    # The conditional edge for the loop
    workflow.add_conditional_edges(
        "store_result",
        should_continue,
        {
            "continue": "generate_topic",
            "end": END
        }
    )
    return workflow.compile()

def main():
    parser = argparse.ArgumentParser(description="Generate a dataset of questions and answers.")
    parser.add_argument("--num-questions", type=int, default=3, help="Number of question-answer pairs to generate.")
    args = parser.parse_args()

    load_dotenv()

    # --- LLM Initialization for Azure OpenAI ---
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        temperature=0.9,
        max_retries=2,
    )
    # -----------------------------------------

    app = build_graph()

    # --- Graph Visualization ---
    try:
        png_data = app.get_graph().draw_mermaid_png()
        with open("workflow_graph.png", "wb") as f:
            f.write(png_data)
        print("--- Generated workflow_graph.png ---")
    except Exception as e:
        print(f"Error generating graph image: {e}")
    # -------------------------

    initial_state = {
        "llm": llm,
        "num_generated": 0,
        "max_generations": args.num_questions,
        "all_results": [],
        "used_topics": []
    }

    # Define a configuration with a dynamic recursion limit
    recursion_limit = args.num_questions * 10  # Allow 10 steps per question
    config = {"recursion_limit": recursion_limit}

    print(f"---STARTING GRAPH EXECUTION for {args.num_questions} questions (recursion limit: {recursion_limit})---")
    final_state = app.invoke(initial_state, config=config)
    print("---GRAPH EXECUTION FINISHED---")

    # Define the output directory and file
    output_dir = "documents"
    output_file = os.path.join(output_dir, "generated_dataset_A.json")

    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_state["all_results"], f, indent=4, ensure_ascii=False)
        print(f"\nSuccessfully saved {len(final_state['all_results'])} question-answer pairs to {output_file}")
    except IOError as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    main()