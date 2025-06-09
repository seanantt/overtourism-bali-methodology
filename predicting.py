import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained("crowded_uncrowded_model")
tokenizer = BertTokenizer.from_pretrained("crowded_uncrowded_model")

# Set the model to evaluation mode
model.eval()

# Load your 200k dataset
data = pd.read_excel("cleaned_data2.xlsx")  # Replace with your file path

# Ensure all texts are strings
data['cleaned_text'] = data['cleaned_text'].astype(str)

# Define a custom dataset for efficient batch processing
class ReviewDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encodings = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        return encodings

# Create a dataset and dataloader for batch processing
dataset = ReviewDataset(data['cleaned_text'].tolist(), tokenizer)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Run predictions
predictions = []
with torch.no_grad():
    for batch in tqdm(dataloader, desc="Predicting", unit="batch"):
        # Reshape the batch data for model input
        input_ids = batch['input_ids'].squeeze(1)  # Remove extra dimension
        attention_mask = batch['attention_mask'].squeeze(1)
        
        # Run the model on the batch
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get the predicted class (0 = Uncrowded, 1 = Crowded)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        
        # Collect predictions
        predictions.extend(preds)

# Add predictions back to the dataframe
data['crowded_label'] = predictions

# Save the result to a new file
data.to_excel("predicted_reviews2.xlsx", index=False)

print("Predictions completed and saved to 'predicted_200k_reviews.xlsx'")
