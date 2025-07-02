# Router for Adaptive-RAG: Autonomous Question-Answer Dataset Generation

This project implements the "Adaptive-RAG" methodology, creating a fully autonomous agent that generates high-quality, diverse question-answer datasets for training and evaluating Retrieval-Augmented Generation (RAG) systems. The agent uses a `LangGraph`-powered workflow to intelligently select question types, fetch content, and produce structured output.

Inspired by the paper [Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity (2403.14403v2)](https://arxiv.org/abs/2403.14403), this implementation aims to build a practical tool for generating the kind of varied-complexity data required for robust RAG evaluation.

## Project Overview

The core of this project is an autonomous agent that performs the following steps in a loop:
1.  **Topic Discovery:** Generates a random topic to explore.
2.  **Content Fetching:** Searches for and retrieves a relevant article from Wikipedia.
3.  **Intelligent Classification:** Analyzes the fetched content to determine the most suitable type of question to generate.
4.  **Question & Answer Generation:** Creates a question-answer pair based on the classification.
5.  **Structured Output:** Saves the generated data to a JSON file, ready for use.

## Question Types

The agent is designed to generate three distinct types of questions to ensure a diverse and challenging dataset:

*   **Type A: Factual Question:** A straightforward question whose answer can be directly found within a single provided text chunk.
    *   *Example:* "What is the primary function of the mitochondria?"
*   **Type B: No-Context Question:** A question on the same topic but whose answer is *not* in the provided text. This tests the model's ability to recognize when it lacks information (hallucination resistance).
    *   *Example:* "Who first discovered the process of cellular respiration?" (When the text only describes the steps of the process).
*   **Type C: Comparative Question:** A more complex question that requires synthesizing information from two different text chunks to formulate an answer.
    *   *Example:* "Compare the economic policies of post-war Japan with those of modern-day Germany."

## Architecture

The agent's workflow is managed by **LangGraph**, a library for building stateful, multi-actor applications with LLMs. The graph defines the sequence of operations, handling everything from topic generation to final output.

The project is divided into three main phases:
1.  **Phase 1 (Completed):** Initial setup of the autonomous generation loop for Type A questions.
2.  **Phase 2 (In Progress):** Implementation of an LLM-based classifier to intelligently choose between Type A, B, and C questions, thereby generating a labeled dataset.
3.  **Phase 3 (Upcoming):** Training a custom, efficient PyTorch-based classifier on the generated data to replace the LLM-based router for production use.

## How to Use

### Prerequisites
- Python 3.8+
- An Azure OpenAI API key with access to a model like `gpt-4o-mini`.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/adaptive-rag.git
    cd adaptive-rag
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file will be added in a future step. For now, manually install `langchain`, `langgraph`, `azure-ai-textanalytics`, `wikipedia`, etc.)*

3.  **Set up your environment:**
    - Create a `.env` file in the root directory.
    - Add your Azure OpenAI credentials to the `.env` file:
      ```
      AZURE_OPENAI_API_KEY="your_api_key"
      AZURE_OPENAI_ENDPOINT="your_endpoint"
      AZURE_OPENAI_DEPLOYMENT_NAME="your_deployment_name"
      AZURE_OPENAI_API_VERSION="api_version"
      ```

### Running the Generator

The main script for generating data is `advanced_question_generator.py`.

```bash
python advanced_question_generator.py --num-questions 10 --output-file documents/generated_dataset.json
```

-   `--num-questions`: The total number of question-answer pairs to generate.
-   `--output-file`: The path to the JSON file where the results will be saved. The script will append to this file if it already exists.

## Project Status

This project is actively under development. See `TODO.md` for the most up-to-date plan and `aididthis.txt` for a log of completed work.
