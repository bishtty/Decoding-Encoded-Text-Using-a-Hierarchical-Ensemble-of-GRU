import pandas as pd
import torch
# Define the model architecture (ensure it matches the saved architecture)
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
        pass

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Seq2Seq(len(char_to_int), 256, 512)
model.load_state_dict(torch.load('seq2seq_model.pth'))
model.to(device)
model.eval()
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
# Assuming the CSV has columns 'no.' and 'string'
df_encoded = pd.read_csv('encoded_strings.csv')
`

### Step 5: Decode the Strings

decoded_results = []
for index, row in df_encoded.iterrows():
    decoded_string = decode(row['string'], model, char_to_int, int_to_char, device)
    decoded_results.append({'no.': row['no.'], 'string': decoded_string})
df_decoded = pd.DataFrame(decoded_results)
df_decoded.to_csv('decoded_strings.csv', index=False)
