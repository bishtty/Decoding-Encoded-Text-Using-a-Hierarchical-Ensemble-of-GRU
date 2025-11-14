import pandas as pd

# Load the datasets
data0 = pd.read_csv('data0.csv')
data1 = pd.read_csv('data1.csv')

# Assuming 'string' columns contain the text
plain_text = data0['string']
encoded_text = data1['string']
def create_vocab(text_series):
    vocab = set(''.join(text_series))
    vocab.add('<PAD>')  # Padding
    vocab.add('<EOS>')  # End of Sentence
    char_to_int = {char: idx for idx, char in enumerate(sorted(vocab))}
    int_to_char = {idx: char for char, idx in char_to_int.items()}
    return char_to_int, int_to_char

char_to_int, int_to_char = create_vocab(pd.concat([plain_text, encoded_text]))
assert '<EOS>' in char_to_int, "The EOS token must be in the vocabulary"

import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, encoded, plain, char_to_int):
        self.encoded = encoded
        self.plain = plain
        self.char_to_int = char_to_int

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        # Convert character to indices for encoded text
        encoded_seq = [self.char_to_int[char] for char in self.encoded.iloc[idx]]
        
        # Convert character to indices for plain text, and append the EOS token
        plain_seq = [self.char_to_int[char] for char in self.plain.iloc[idx]]
        plain_seq.append(self.char_to_int['<EOS>'])  # adding <EOS> as a separate token

        return torch.tensor(encoded_seq), torch.tensor(plain_seq)
def collate_batch(batch):
    # Unzip the batch data
    encoded_seqs, plain_seqs = zip(*batch)
    
    # Compute max lengths for zero padding
    max_encoded_length = max([seq.size(0) for seq in encoded_seqs])
    max_plain_length = max([seq.size(0) for seq in plain_seqs])
    
    # Prepare padded batches
    encoded_batch = torch.zeros(len(encoded_seqs), max_encoded_length, dtype=torch.long)
    plain_batch = torch.zeros(len(plain_seqs), max_plain_length, dtype=torch.long)
    
    # Fill up the tensors with padded data
    for i, (enc_seq, pln_seq) in enumerate(zip(encoded_seqs, plain_seqs)):
        encoded_batch[i, :enc_seq.size(0)] = enc_seq
        plain_batch[i, :pln_seq.size(0)] = pln_seq

    return encoded_batch, plain_batch

dataset = TextDataset(encoded_text, plain_text, char_to_int)
loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_batch)
import torch.nn as nn

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Seq2Seq, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.decoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, src, tgt):
        embedded_src = self.embedding(src)
        _, hidden = self.encoder(embedded_src)

        embedded_tgt = self.embedding(tgt)
        output, _ = self.decoder(embedded_tgt, hidden)
        return self.fc(output)

model = Seq2Seq(len(char_to_int), 256, 512)
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = Adam(model.parameters())
criterion = CrossEntropyLoss(ignore_index=char_to_int['<PAD>'])

def train(model, loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        for encoded, plain in loader:
            encoded, plain = encoded.to(device), plain.to(device)
            optimizer.zero_grad()
            output = model(encoded, plain[:, :-1])
            loss = criterion(output.reshape(-1, output.shape[2]), plain[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item()}")

train(model, loader, optimizer, criterion, 20)
torch.save(model.state_dict(), 'seq2seq_model.pth')
def decode(encoded_text, model, char_to_int, int_to_char, device):
    model.eval()
    with torch.no_grad():
        # Convert the input string to a sequence of indices
        encoded_seq = torch.tensor([char_to_int[char] for char in encoded_text]).unsqueeze(0).to(device)
        
        # Initialize the tensor for storing the output indices
        outputs = []
        
        # Predict the initial hidden state with the encoder
        embedded = model.embedding(encoded_seq)
        _, hidden = model.encoder(embedded)
        
        # Start the decoding process, assume the start token is the first character of the encoded text
        input = torch.tensor([char_to_int[encoded_text[0]]], device=device)

        while True:
            # Predict the next token
            embedded = model.embedding(input.unsqueeze(0))
            output, hidden = model.decoder(embedded, hidden)
            output = model.fc(output.squeeze(0))
            
            # Find the character with the highest probability
            predicted = output.argmax(1).item()
            outputs.append(predicted)
            
            # Break if the EOS token was predicted
            if int_to_char[predicted] == '<EOS>':
                break
            
            # Update the input for the next prediction
            input = torch.tensor([predicted], device=device)

        # Convert the list of indices to a string
        decoded_text = ''.join([int_to_char[idx] for idx in outputs if int_to_char[idx] != '<EOS>'])
        return decoded_text

# Example of decoding
decoded_text = decode("YMJ RTRJSY FX XYFWYQJI FX N MNX MFSI HQTXJI QNPJ", model, char_to_int, int_to_char, device)
print(decoded_text)
