import torch
import gradio as gr
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# --- Configuration ---
MODEL_PATH = 'models/question_classifier.pth'
MODEL_NAME = 'distilbert-base-uncased'

# --- Load Model and Tokenizer ---
print("Loading model and tokenizer...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

# Load model architecture
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3  # A, B, C
)

# Load the fine-tuned weights
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    print("Please ensure you have trained the model by running 'python train_classifier.py' first.")
    exit()

model.to(device)
model.eval()
print("Model and tokenizer loaded successfully.")

# --- Prediction Function ---
def classify_question(question):
    """
    Classifies a given question as Type A, B, or C using the trained model.
    """
    if not question.strip():
        return "Please enter a question."

    # Tokenize the input question
    inputs = tokenizer(
        question,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)

    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction_idx = torch.argmax(logits, dim=-1).item()

    # Map index to label
    label_map = {0: 'Type A (Factual)', 1: 'Type B (No-Context)', 2: 'Type C (Comparative)'}
    predicted_label = label_map.get(prediction_idx, "Unknown")

    return predicted_label

# --- Gradio Interface ---
iface = gr.Interface(
    fn=classify_question,
    inputs=gr.Textbox(lines=3, label="Enter a question to classify:", placeholder="e.g., 'What is the capital of France?' or 'Compare the works of Shakespeare and Marlowe.'"),
    outputs=gr.Label(label="Predicted Question Type"),
    title="Adaptive-RAG Question Classifier",
    description="Enter a question to see how the trained DistilBERT model classifies it. This helps validate the model that will be used to route questions in the autonomous RAG dataset generator.",
    examples=[
        ["What is the main ingredient in guacamole?"],
        ["Who was the first person to walk on the moon?"],
        ["What are the differences between Python lists and tuples?"],
        ["Explain the impact of the printing press on the Renaissance."]
    ]
)

if __name__ == "__main__":
    print("Launching Gradio UI...")
    iface.launch()
