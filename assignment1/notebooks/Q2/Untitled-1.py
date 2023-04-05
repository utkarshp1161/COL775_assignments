# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import pandas as pd
from sklearn.model_selection import train_test_split
from torchtext.vocab import GloVe
from torch.utils.data import DataLoader
from tqdm import tqdm

# %%
df_train = pd.read_csv("../../data/Text-To-SQL-COL775/train.csv")
df_tables = pd.read_json("../../data/Text-To-SQL-COL775/tables.json")

# %%
df_tables

# %%
df_train

# %%
data_file = "../../data/Text-To-SQL-COL775/train.csv"
table_file = "../../data/Text-To-SQL-COL775/tables.json"

class SpiderDataset(data.Dataset):
    def __init__(self, data_file, table_file):
        self.data = pd.read_csv(data_file)
        self.table_data = pd.read_json(table_file)
        self.vocab = GloVe(name='6B', dim=100)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        question = self.data.iloc[index]['question'] # question extracted
        query = self.data.iloc[index]['query'] # query extracted
        schema = self.data.iloc[index]['db_id'] # get database id
        # Get the table and column names for the database schema
        tables = self.table_data[self.table_data['db_id'] == schema]['table_names_original'].tolist() # entire table data for that db_id
        columns = self.table_data[self.table_data['db_id'] == schema]['column_names_original'].tolist()
        # Concatenate the table and column names to the question and query inputs
        question += ' ' + ' '.join(tables) + ' ' + ' '.join(columns)
        query += ' ' + ' '.join(tables) + ' ' + ' '.join(columns)
        return question, query, schema
    
    def collate_fn(self, batch):
        questions, queries, schemas = zip(*batch)
        # Tokenize and convert to tensors
        questions = [torch.tensor(self.vocab.stoi[token] for token in question.split()) for question in questions]
        queries = [torch.tensor(self.vocab.stoi[token] for token in query.split()) for query in queries]
        schemas = [torch.tensor(self.vocab.stoi[token] for token in schema.split()) for schema in schemas]
        # Pad sequences to the same length
        questions = nn.utils.rnn.pad_sequence(questions, batch_first=True)
        queries = nn.utils.rnn.pad_sequence(queries, batch_first=True)
        schemas = nn.utils.rnn.pad_sequence(schemas, batch_first=True)
        return questions, queries, schemas


# %%
dataset = SpiderDataset('data_file.csv', 'table_file.json')

class Seq2SeqLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(Seq2SeqLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.decoder = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, input_seq, target_seq, input_lengths):
        embedded_input = self.embedding(input_seq)
        packed_input = pack_padded_sequence(embedded_input, input_lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.encoder(packed_input)
        encoder_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        decoder_output, _ = self.decoder(embedded_input, hidden)
        output = self.linear(decoder_output)
        return output


# Define hyperparameters
vocab_size = len(dataset.vocab)
embedding_dim = 100
hidden_dim = 256
num_layers = 2
learning_rate = 0.001
batch_size = 64
num_epochs = 10
model = Seq2SeqLSTM(vocab_size, embedding_dim, hidden_dim, num_layers)
# Define the model and optimizer


model = Seq2SeqLSTM(vocab_size, embedding_dim, hidden_dim, num_layers)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Define the loss function
loss_fn = nn.CrossEntropyLoss()
# Split the dataset into train, validation, and test sets
train_size = int(len(dataset) * 0.8)
val_size = int(len(dataset) * 0.1)
test_size = len(dataset) - train_size - val_size
train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# Create data loaders for the train, validation, and test sets
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)

# Train the model
for epoch in range(num_epochs):
    train_loss = 0.0
    val_loss = 0.0
    model.train()
    for i, batch in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        questions, queries, _ = batch
        input_lengths = [len(seq) for seq in questions]
        target_lengths = [len(seq) for seq in queries]
        output = model(questions, queries[:,:-1], input_lengths)
        target = queries[:,1:]
        loss = loss_fn(output.reshape(-1, vocab_size), target.reshape(-1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            questions, queries, _ = batch
            input_lengths = [len(seq) for seq in questions]
            target_lengths = [len(seq) for seq in queries]
            output = model(questions, queries[:,:-1], input_lengths)
            target = queries[:,1:]
            loss = loss_fn(output.reshape(-1, vocab_size), target.reshape(-1))
            val_loss += loss.item()
        val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")