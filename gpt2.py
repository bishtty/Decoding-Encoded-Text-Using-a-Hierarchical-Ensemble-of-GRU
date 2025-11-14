import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F

# Step 1: Load the data
encoded_df = pd.read_csv('encoded.csv')
clean_df = pd.read_csv('testing.csv')

# Step 2: Prepare the data
class TextCorrectionDataset(Dataset):
    def __init__(self, encoded_texts, clean_texts, tokenizer, max_length=512):
        self.encoded_texts = encoded_texts
        self.clean_texts = clean_texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.encoded_texts)

    def __getitem__(self, idx):
        encoded_text = self.encoded_texts[idx]
        clean_text = self.clean_texts[idx]
        inputs = self.tokenizer.encode_plus(
            encoded_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        targets = self.tokenizer.encode_plus(
            clean_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        target_ids = targets['input_ids'].squeeze()
        return input_ids, attention_mask, target_ids

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set the padding token and padding side
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

# Create the dataset and dataloader
dataset = TextCorrectionDataset(encoded_df['string'], clean_df['string'], tokenizer)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Step 3: Define the model
class GPT2CorrectionModel(torch.nn.Module):
    def __init__(self, pretrained_model_name='gpt2'):
        super(GPT2CorrectionModel, self).__init__()
        self.model = GPT2LMHeadModel.from_pretrained(pretrained_model_name)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss, outputs.logits

model = GPT2CorrectionModel()
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Step 4: Train the model
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

def train(model, dataloader, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            optimizer.zero_grad()
            loss, logits = model(input_ids, attention_mask, labels=labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train(model, dataloader, optimizer, device, epochs=10)

# Step 5: Save the model
model_path = 'text_correction_model.pth'
torch.save(model.state_dict(), model_path)
tokenizer.save_pretrained('./')

print(f"Model saved to {model_path}")

# Example function to use the model for correction
def correct_text(model, tokenizer, encoded_text, device):
    model.eval()
    inputs = tokenizer.encode_plus(
        encoded_text,
        add_special_tokens=True,
        return_tensors="pt",
        padding='max_length',
        truncation=True,
        max_length=512
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50,  # Adjust this value if needed
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

# Load the trained model for inference
model = GPT2CorrectionModel()
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)

# Correct a sample encoded text
sample_encoded_text = "LE NALL SO LONG WAS HE THAT MR HOLDER AND I WEN"
corrected_text = correct_text(model, tokenizer, sample_encoded_text, device)
print("Corrected text:", corrected_text)
