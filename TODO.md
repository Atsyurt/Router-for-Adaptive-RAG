# Project: Fully Autonomous Question Generation Agent with LangGraph

This file outlines the revised plan for an agent that autonomously finds topics, fetches content, and generates question/answer pairs for creating a RAG dataset.

## Collaboration Model
*   **Your Role is Code Generation:** Your primary function is to write the Python code for our project, primarily in `intelligent_question_generator.py` and later `train_classifier.py`.
*   **User (Your Role):** Provide direction, manage the environment, and execute all code.

## Revised Step-by-Step Project Plan

The project uses `LangGraph` to manage the workflow.

---

### **Phase 1: Autonomous Type A Question & Answer Generation (COMPLETED)**
*Goal: Create a graph that can, from scratch, find a topic, fetch an article, and generate a simple question and its corresponding answer based on the text.*

- [x] **Step 1.1 (AI):** Define the `LangGraph` state schema.
- [x] **Step 1.2 (AI):** Implement the "Topic Generation Node."
- [x] **Step 1.3 (AI):** Implement the "Wikipedia Search Node."
- [x] **Step 1.4 (AI):** Implement the "Content Fetching Node."
- [x] **Step 1.5 (AI):** Implement the "Type A Question Generation Node."
- [x] **Step 1.6 (AI):** Implement the "Answer Generation Node."
- [x] **Step 1.7 (AI):** Implement Command-Line Arguments (`argparse`).
- [x] **Step 1.8 (AI):** Implement a Master Loop via conditional graph edges.
- [x] **Step 1.9 (AI):** Assemble and compile the full graph.
- [x] **Step 1.10 (User):** Successfully run the script and generate `generated_dataset_A.json`.
- [x] **Step 1.11 (AI):** Add robustness features: topic repetition avoidance, API quota delay, and recursion limit handling.
- [x] **Step 1.12 (AI):** Switch to Azure OpenAI to resolve Google API quota issues.

---

### **Phase 2: Intelligent Question Classification & Generation (IN PROGRESS)**
*Goal: Replace the random question type selection with an intelligent, two-stage classification system.*

- [x] **Step 2.1 (AI):** Implement initial Type B (No-Context) and Type C (Comparative) question generation nodes.
- [x] **Step 2.2 (AI):** Implement a basic random classifier to enable end-to-end generation of all three question types.
- [x] **Step 2.3 (AI):** Refine file I/O to append new questions to existing JSON datasets instead of overwriting them.
- [x] **Step 2.4 (AI):** Improve robustness by adding a validity check after content fetching to handle failed lookups and prevent crashes.
- [x] **Step 2.5 (AI):** Refine Type C generation to use and save shorter, fixed-size text chunks (3000 chars) for consistency.
- [x] All of these step 2 implemented to the advanced_question_generator.py file until now
- [x] **Step 2.6 (AI):** **Implement LLM-based "Zero-Shot" Classifier.**
    - Upgrade the `classify_question` node to use the main LLM.
    - The LLM will analyze the document and determine the most suitable question type (A, B, or C).
    - This step is crucial for generating the **labeled dataset** required for training our own model.
- [x] **Step 2.7 (User):** Run the generator script to produce a labeled dataset of at least a few hundred examples.

---

### **Phase 3: Custom PyTorch Classifier for Routing**
*Goal: Train a small, fast, and efficient Transformer-based classifier using pure PyTorch and integrate it into the graph for production-level routing.*

**CRITICAL REMINDER:** The classifier's purpose is to predict question complexity based *only on the question text itself*, not the source document. This aligns with the Adaptive-RAG paper's methodology, where the classifier acts as a router *before* content is fetched or processed. All future implementations must adhere to this principle.

- [x] **Step 3.1 (AI):** **Create `train_classifier.py` Script.**
    - This script will be written using pure PyTorch.
    - It will define a custom `Dataset` to load our labeled JSON data.
    - It will define a custom `nn.Module` class that wraps a pre-trained Hugging Face model (e.g., DistilBERT) and adds a classification head.
    - It will contain a from-scratch training loop to fine-tune the model on our data.
- [x] **Step 3.2 (User):** Execute the `train_classifier.py` script to train and save the custom model.
- [ ] **Step 3.3 (AI):** **Integrate the Trained Model.**
    - Create a new node, `route_with_pytorch_classifier`, in a new `.py` file.
    - This node will load our trained PyTorch model.
    - It will perform inference locally and efficiently to choose the question route.
- [ ] **Step 3.4 (AI):** Update the graph to use the new PyTorch-based routing node instead of the LLM-based one.
- [ ] **Step 3.5 (User):** Final testing of the fully autonomous agent with the integrated high-performance classifier.
- [ ] **Step 3.6 (AI):** **Create `test_ui.py` for Classifier Validation.**
    - Build a simple web UI using Gradio.
    - The UI will take a question as input.
    - It will use the trained PyTorch model to classify the question as Type A, B, or C and display the prediction.
    - This provides a quick way to interactively test and validate the classifier's performance.