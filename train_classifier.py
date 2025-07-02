import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
MODEL_NAME = 'distilbert-base-uncased'
DATA_DIR = 'documents'
MODEL_SAVE_PATH = 'models/question_classifier.pth'
NUM_EPOCHS = 10 # Increased epochs for better training
BATCH_SIZE = 16 # Adjusted for potentially larger datasets
LEARNING_RATE = 5e-5
TEST_SIZE = 0.2

# --- 1. Load and Prepare Data ---

def load_and_merge_data(data_dir):
    """Loads and merges JSON data from the specified directory."""
    all_data = []
    print(f"Scanning for JSON files in '{data_dir}'...")
    for filename in os.listdir(data_dir):
        if filename.startswith('generated_dataset_') and filename.endswith('.json'):
            file_path = os.path.join(data_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    question_type = filename.split('_')[-1].split('.')[0].upper()
                    for item in data:
                        item['question_type'] = question_type
                    all_data.extend(data)
                print(f"Successfully loaded and processed {filename}")
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not read or parse {filename}. Skipping. Error: {e}")
    return all_data

def prepare_data(raw_data):
    """Prepares questions and labels for the model."""
    texts = []
    labels = []
    label_map = {'A': 0, 'B': 1, 'C': 2}

    for item in raw_data:
        question = item.get('question')
        q_type = item.get('question_type')
        if question and q_type in label_map:
            texts.append(question)
            labels.append(label_map[q_type])
        else:
            print(f"Warning: Skipping item due to missing question or invalid type: {q_type}")
    print(f"Prepared {len(texts)} question-label pairs.")
    return texts, labels

# --- 2. Custom PyTorch Dataset ---

class QuestionClassifierDataset(Dataset):
    """Custom dataset for the question classifier."""
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_len,
            return_token_type_ids=False, padding='max_length',
            truncation=True, return_attention_mask=True, return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# --- 3. Training and Evaluation ---

def train_epoch(model, data_loader, loss_fn, optimizer, device):
    """Trains the model for one epoch."""
    model = model.train()
    losses = []
    for d in data_loader:
        input_ids, attention_mask, labels = d["input_ids"].to(device), d["attention_mask"].to(device), d["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return np.mean(losses)

def eval_model(model, data_loader, device):
    """Evaluates the model and returns accuracy, predictions, and actual labels."""
    model = model.eval()
    predictions, actual_labels = [], []
    with torch.no_grad():
        for d in data_loader:
            input_ids, attention_mask, labels = d["input_ids"].to(device), d["attention_mask"].to(device), d["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            predictions.extend(preds)
            actual_labels.extend(labels)
    predictions = torch.stack(predictions).cpu()
    actual_labels = torch.stack(actual_labels).cpu()
    accuracy = accuracy_score(actual_labels, predictions)
    return accuracy, predictions, actual_labels

# --- 4. Plotting ---

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plots a confusion matrix using seaborn."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix of the Best Model')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

# --- Main Execution ---

if __name__ == '__main__':
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    raw_data = load_and_merge_data(DATA_DIR)
    if not raw_data:
        print("No data loaded. Exiting.")
        exit()

    texts, labels = prepare_data(raw_data)
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=TEST_SIZE, random_state=42, stratify=labels
    )
    train_dataset = QuestionClassifierDataset(train_texts, train_labels, tokenizer)
    val_dataset = QuestionClassifierDataset(val_texts, val_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    print(f"Starting training on {device}...")
    print("-" * 30)

    best_accuracy = 0
    best_preds, best_labels = None, None

    for epoch in range(NUM_EPOCHS):
        print(f'Epoch {epoch + 1}/{NUM_EPOCHS}')
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        print(f'Training loss: {train_loss}')
        val_acc, val_preds, val_labels_epoch = eval_model(model, val_loader, device)
        print(f'Validation Accuracy: {val_acc}')
        print("-" * 30)

        if val_acc > best_accuracy:
            print(f"Accuracy improved. Saving model to {MODEL_SAVE_PATH}")
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            best_accuracy = val_acc
            # Store the best predictions and their corresponding true labels
            best_preds = val_preds
            best_labels = val_labels_epoch


    print("Training complete.")
    print(f"Best validation accuracy: {best_accuracy}")
    print(f"Model saved to {MODEL_SAVE_PATH}")

    if best_preds is not None and best_labels is not None:
        print("Displaying confusion matrix for the best model...")
        class_names = ['Type A', 'Type B', 'Type C']
        plot_confusion_matrix(best_labels, best_preds, class_names)