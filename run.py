import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Function to create vocabulary
def create_vocab(text_series):
    vocab = set(''.join(text_series))
    vocab.add('<PAD>')  # Padding
    vocab.add('<EOS>')  # End of Sentence
    char_to_int = {char: idx for idx, char in enumerate(sorted(vocab))}
    int_to_char = {idx: char for char, idx in char_to_int.items()}
    return char_to_int, int_to_char

# Assuming 'string' columns contain the text
data0 = pd.read_csv('data0.csv')
data1 = pd.read_csv('data1.csv')
plain_text = data0['string']
encoded_text = data1['string']
char_to_int, int_to_char = create_vocab(pd.concat([plain_text, encoded_text]))

# Define the Seq2Seq model
class Seq2Seq(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Seq2Seq, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.encoder = torch.nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.decoder = torch.nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, src, tgt):
        embedded_src = self.embedding(src)
        _, hidden = self.encoder(embedded_src)
        embedded_tgt = self.embedding(tgt)
        output, _ = self.decoder(embedded_tgt, hidden)
        return self.fc(output)

# Load pre-trained Seq2Seq model
seq2seq_model = Seq2Seq(len(char_to_int), 256, 512)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq2seq_model.load_state_dict(torch.load('seq2seq_model.pth', map_location=device))
seq2seq_model.to(device)

# Function to decode encoded message
def decode(encoded_text, model, char_to_int, int_to_char, device):
    model.eval()
    with torch.no_grad():
        encoded_seq = torch.tensor([char_to_int[char] for char in encoded_text]).unsqueeze(0).to(device)
        outputs = []
        embedded = model.embedding(encoded_seq)
        _, hidden = model.encoder(embedded)
        input = torch.tensor([char_to_int[encoded_text[0]]], device=device)
        while True:
            embedded = model.embedding(input.unsqueeze(0))
            output, hidden = model.decoder(embedded, hidden)
            output = model.fc(output.squeeze(0))
            predicted = output.argmax(1).item()
            outputs.append(predicted)
            if int_to_char[predicted] == '<EOS>':
                break
            input = torch.tensor([predicted], device=device)
        decoded_text = ''.join([int_to_char[idx] for idx in outputs if int_to_char[idx] != '<EOS>'])
        return decoded_text

# Example of decoding
encoded_message = "PSTB BMD STY XFNI YMJ NSXUJHYTW MJ ITJXSY QTT"
decoded_text = decode(encoded_message, seq2seq_model, char_to_int, int_to_char, device)
print("Decoded text:", decoded_text)

# Define the GPT-2 Text Correction model
class GPT2CorrectionModel(torch.nn.Module):
    def __init__(self, pretrained_model_name='gpt2'):
        super(GPT2CorrectionModel, self).__init__()
        self.model = GPT2LMHeadModel.from_pretrained(pretrained_model_name)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss, outputs.logits

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

# Load pre-trained Text Correction model
correction_model = GPT2CorrectionModel()
correction_model.load_state_dict(torch.load('text_correction_model.pth', map_location=device))
correction_model = correction_model.to(device)

# Function to correct text
# Function to correct text
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

# Load the trained correction model for inference
correction_model = GPT2CorrectionModel()
correction_model.load_state_dict(torch.load('text_correction_model.pth', map_location=device))
correction_model = correction_model.to(device)

# Correct the decoded text
corrected_text = correct_text(correction_model, tokenizer, decoded_text, device)
print("Corrected text:", corrected_text)
