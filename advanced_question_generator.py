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
    source_chunk_2: str
    llm: BaseChatModel
    num_generated: int
    max_generations: int
    results_A: list # <-- UPDATED
    results_B: list # <-- ADDED
    results_C: list # <-- ADDED
    used_topics: list[str]
    question_type: str

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
        return {"document": None, "source_chunk": None}
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
        
        # Create the source chunk here, before any branching
        source_chunk = text[:4000]
        
        print("Successfully fetched and prepared document and source_chunk.")
        return {"document": text, "source_chunk": source_chunk}
    except requests.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return {"document": None, "source_chunk": None}

def generate_question_A(state):
    """Generates a Type A (factual) question based on the document."""
    print("---NODE: GENERATING QUESTION (TYPE A)---")
    llm = state['llm']
    source_chunk = state['source_chunk'] # Get chunk from state
    if not source_chunk:
        return {"question": None}
        
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
    # We only return the question here, the chunk is already in the state
    return {"question": generated_question.content}

def generate_question_B(state):
    """Generates a Type B (no-context) question."""
    print("---NODE: GENERATING QUESTION (TYPE B)---")
    llm = state['llm']
    topic = state['topic']
    source_chunk = state['source_chunk']

    class QuestionB(BaseModel):
        question: str = Field(description="A question that is highly relevant to the main topic but cannot be answered by the provided context.")

    prompt = PromptTemplate.from_template(
        """Your task is to generate a single question about a specific topic that CANNOT be answered from the given text context.
        The question must be specific and relevant to the overall topic, but the answer should not be present in the context.

        TOPIC: {topic}

        CONTEXT:
        ---
        {context}
        ---

        Based on the topic and the context, generate a question that a user might ask but which you know is not answered here.
        """
    )
    
    question_gen_chain = prompt | llm.with_structured_output(QuestionB)
    generated_question = question_gen_chain.invoke({"topic": topic, "context": source_chunk})
    
    answer = "The provided text does not contain the information needed to answer this question."
    
    print(f"Generated Question (Type B): {generated_question.question}")
    return {"question": generated_question.question, "answer": answer, "question_type": "Type B"}

def generate_question_C(state):
    """Generates a Type C (comparative) question by synthesizing info from two different chunks."""
    print("---NODE: GENERATING QUESTION (TYPE C)---")
    llm = state['llm']
    document = state['document']
    
    # Define the character limit for the chunks
    max_chunk_size = 3000
    
    # This node is now only called when the document is long enough.
    # We can safely split it into two halves.
    midpoint = len(document) // 2
    chunk1_full = document[:midpoint]
    chunk2_full = document[midpoint:]

    # Truncate chunks to the desired size for both LLM and saving
    chunk1_for_processing = chunk1_full[:max_chunk_size]
    chunk2_for_processing = chunk2_full[:max_chunk_size]

    class QuestionC(BaseModel):
        question: str = Field(description="A complex question that requires synthesizing information from both provided text chunks.")
        answer: str = Field(description="A concise answer to the question, derived by combining information from both text chunks.")

    prompt = PromptTemplate.from_template(
        """Your task is to act as a research analyst. You have been given two distinct chunks of text from a larger document.
        Your goal is to create a single, insightful question that requires a reader to connect or compare information found in **both** Chunk 1 and Chunk 2.
        After formulating the question, provide a comprehensive answer, also synthesized from the information in both chunks.

        CONTEXT CHUNK 1:
        ---
        {chunk1}
        ---

        CONTEXT CHUNK 2:
        ---
        {chunk2}
        ---

        Generate the complex, synthesis-based question and its corresponding answer.
        """
    )
    
    qa_gen_chain = prompt | llm.with_structured_output(QuestionC)
    # Use the truncated chunks for the LLM call
    generated_qa = qa_gen_chain.invoke({"chunk1": chunk1_for_processing, "chunk2": chunk2_for_processing})
    
    print(f"Generated Question (Type C): {generated_qa.question}")
    print(f"Generated Answer (Type C): {generated_qa.answer}")
    
    # Return the truncated chunks for logging, ensuring they are the same ones used for generation
    return {
        "question": generated_qa.question, 
        "answer": generated_qa.answer, 
        "question_type": "Type C",
        "source_chunk": chunk1_for_processing,
        "source_chunk_2": chunk2_for_processing
    }

def classify_question(state):
    """
    Classifies which type of question to generate next based on document length.
    """
    print("---NODE: CLASSIFYING QUESTION TYPE---")
    document = state.get("document", "")
    
    # Define possible choices
    choices = ["A", "B"]
    
    # Only allow Type C if the document is long enough for a meaningful split
    if len(document) > 4000: # A reasonable minimum length for two chunks
        choices.append("C")
    
    import random
    question_type = random.choice(choices)
    print(f"  - Document length: {len(document)} chars.")
    print(f"  - Available choices: {choices}")
    print(f"  - Decision: Generate a Type {question_type} question.")
    return {"question_type": f"Type {question_type}"}


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
    """Stores the result in the correct list based on question type."""
    print("---NODE: STORING RESULT---")
    q_type = state["question_type"]
    
    # Create the result dictionary
    result = {
        "topic": state["topic"],
        "url": state["url"],
        "question": state["question"],
        "answer": state["answer"],
        "question_type": q_type,
        "source_chunk": state.get("source_chunk", ""),
        "source_chunk_2": state.get("source_chunk_2", "") # Add the second chunk if it exists
    }
    
    # Get current state lists
    results_A = state.get("results_A", [])
    results_B = state.get("results_B", [])
    results_C = state.get("results_C", [])
    
    # Append to the correct list
    if q_type == "Type A":
        results_A.append(result)
    elif q_type == "Type B":
        results_B.append(result)
    elif q_type == "Type C":
        results_C.append(result)
        
    # Update state
    updated_count = state["num_generated"] + 1
    updated_used_topics = state["used_topics"] + [state["topic"]]
    
    return {
        "results_A": results_A,
        "results_B": results_B,
        "results_C": results_C,
        "num_generated": updated_count,
        "used_topics": updated_used_topics,
        "question": None, # Clear out previous run's data
        "answer": None,
        "source_chunk": None,
        "source_chunk_2": None,
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
        delay = 3 # seconds
        print(f"  Decision: CONTINUE to next loop after a {delay}-second delay.")
        time.sleep(delay)
        return "continue"

def check_document_validity(state):
    """
    Conditional edge: checks if a valid document was fetched.
    If not, it routes back to the beginning to try a new topic.
    """
    print("---EDGE: CHECKING DOCUMENT VALIDITY---")
    document = state.get("document")
    if document:
        print("  - Document is valid. Proceeding to question classification.")
        return "continue"
    else:
        print("  - Document is NOT valid (None). Skipping and generating a new topic.")
        # Optional: Add a small delay here as well if needed
        # time.sleep(1) 
        return "retry"

# --- 4. Build the Graph ---

def build_graph():
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("generate_topic", generate_topic)
    workflow.add_node("search_wikipedia", search_wikipedia)
    workflow.add_node("fetch_content", fetch_content)
    workflow.add_node("classify_question", classify_question)
    workflow.add_node("generate_question_A", generate_question_A)
    workflow.add_node("generate_question_B", generate_question_B)
    workflow.add_node("generate_question_C", generate_question_C)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("store_result", store_result)

    # Set entry point and basic flow
    workflow.set_entry_point("generate_topic")
    workflow.add_edge("generate_topic", "search_wikipedia")
    workflow.add_edge("search_wikipedia", "fetch_content")
    
    # Add the new conditional edge for document validity
    workflow.add_conditional_edges(
        "fetch_content",
        check_document_validity,
        {
            "continue": "classify_question",
            "retry": "generate_topic" # If doc is invalid, loop back to the start
        }
    )

    # Conditional routing based on question type
    workflow.add_conditional_edges(
        "classify_question",
        lambda state: state["question_type"],
        {
            "Type A": "generate_question_A",
            "Type B": "generate_question_B",
            "Type C": "generate_question_C",
        }
    )

    # Path for Type A questions
    workflow.add_edge("generate_question_A", "generate_answer")
    workflow.add_edge("generate_answer", "store_result")

    # Paths for Type B and C questions (they generate their own answers)
    workflow.add_edge("generate_question_B", "store_result")
    workflow.add_edge("generate_question_C", "store_result")
    
    # The final conditional edge for the main loop
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
        "results_A": [],
        "results_B": [],
        "results_C": [],
        "used_topics": []
    }

    # Define a configuration with a dynamic recursion limit
    recursion_limit = args.num_questions * 10  # Allow 10 steps per question
    config = {"recursion_limit": recursion_limit}

    print(f"---STARTING GRAPH EXECUTION for {args.num_questions} questions (recursion limit: {recursion_limit})---")
    final_state = app.invoke(initial_state, config=config)
    print("---GRAPH EXECUTION FINISHED---")

    # --- Save results to separate files (append mode) ---
    output_dir = "documents"
    os.makedirs(output_dir, exist_ok=True)

    datasets = {
        "A": final_state.get("results_A", []),
        "B": final_state.get("results_B", []),
        "C": final_state.get("results_C", []),
    }

    for type_key, data in datasets.items():
        if not data:
            print(f"No new 'Type {type_key}' questions were generated. Skipping file update.")
            continue
            
        output_file = os.path.join(output_dir, f"generated_dataset_{type_key}.json")
        
        # --- Append Logic ---
        try:
            # 1. Read existing data if the file exists
            existing_data = []
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                with open(output_file, "r", encoding="utf-8") as f:
                    try:
                        existing_data = json.load(f)
                        if not isinstance(existing_data, list):
                           print(f"Warning: Existing file {output_file} is not a JSON list. Overwriting.")
                           existing_data = []
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode JSON from {output_file}. Overwriting file.")
                        existing_data = []
            
            # 2. Append new data
            combined_data = existing_data + data
            
            # 3. Write the combined data back to the file
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(combined_data, f, indent=4, ensure_ascii=False)
                
            print(f"Successfully appended {len(data)} 'Type {type_key}' questions to {output_file}. Total questions: {len(combined_data)}.")

        except IOError as e:
            print(f"Error saving file {output_file}: {e}")

if __name__ == "__main__":
    main()