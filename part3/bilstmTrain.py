# bilstmTrain.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import numpy as np
import time
from torch.nn.utils.rnn import pad_sequence

# === Hyperparameters ===
EMBEDDING_DIM = 100
HIDDEN_DIM = 64
EPOCHS = 5
LEARNING_RATE = 0.01
BATCH_SIZE = 64
GPU_NUMBER = 1

# === Dataset Handling ===
class TaggingDataset(Dataset):
    def __init__(self, data_path, word_to_ix, tag_to_ix):
        self.sentences = []
        self.labels = []
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix
        with open(data_path) as f:
            sentence = []
            tags = []
            for line in f:
                line = line.strip()
                if line == "":
                    if sentence:
                        self.sentences.append(torch.tensor(sentence))
                        self.labels.append(torch.tensor(tags))
                        sentence = []
                        tags = []
                else:
                    word, tag = line.split()
                    sentence.append(word_to_ix.get(word, word_to_ix["<UNK>"]))
                    tags.append(tag_to_ix[tag])
            if sentence:
                self.sentences.append(torch.tensor(sentence))
                self.labels.append(torch.tensor(tags))

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]

    # Collate function for DataLoader
    def collate_fn(batch):
        sentences, tags = zip(*batch)
        sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)
        tags_padded = pad_sequence(tags, batch_first=True, padding_value=-1)
        return sentences_padded, tags_padded

# === Model Definition ===
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim):
        super(BiLSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm_fwd = nn.LSTMCell(embedding_dim, hidden_dim)
        self.lstm_bwd = nn.LSTMCell(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, sentence):
        embeds = self.embedding(sentence)
        seq_len = embeds.size(0)

        h_fwd = torch.zeros(1, self.hidden_dim)
        c_fwd = torch.zeros(1, self.hidden_dim)
        h_bwd = torch.zeros(1, self.hidden_dim)
        c_bwd = torch.zeros(1, self.hidden_dim)

        outputs_fwd = []
        for i in range(seq_len):
            h_fwd, c_fwd = self.lstm_fwd(embeds[i].unsqueeze(0), (h_fwd, c_fwd))
            outputs_fwd.append(h_fwd)

        outputs_bwd = []
        for i in reversed(range(seq_len)):
            h_bwd, c_bwd = self.lstm_bwd(embeds[i].unsqueeze(0), (h_bwd, c_bwd))
            outputs_bwd.insert(0, h_bwd)

        outputs = [torch.cat((f, b), dim=1) for f, b in zip(outputs_fwd, outputs_bwd)]
        tag_space = torch.stack([self.hidden2tag(o) for o in outputs])
        return tag_space

# === Utilities ===
def build_vocab(data_path):
    word_to_ix = {"<UNK>": 0}
    tag_to_ix = {}
    with open(data_path) as f:
        for line in f:
            if line.strip() == "":
                continue
            word, tag = line.strip().split()
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)
    return word_to_ix, tag_to_ix

# === Training Procedure ===
def train_model(train_path, dev_path):
    device = torch.device(f"cuda:{GPU_NUMBER}" if torch.cuda.is_available() else "cpu")

    word_to_ix, tag_to_ix = build_vocab(train_path)
    train_data = TaggingDataset(train_path, word_to_ix, tag_to_ix)
    dev_data = TaggingDataset(dev_path, word_to_ix, tag_to_ix)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=TaggingDataset.collate_fn)
    dev_loader = DataLoader(dev_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=TaggingDataset.collate_fn)

    log_file = f"train_log_{"pos " if "pos" in train_path else "ner"}.txt"

    model = BiLSTMTagger(len(word_to_ix), len(tag_to_ix), EMBEDDING_DIM, HIDDEN_DIM).to(device)
    loss_function = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training on {len(train_data)} sentences with {len(word_to_ix)} words\n Starting training...")

    with open(log_file, "w") as log_f:
        for epoch in range(EPOCHS):
            total_loss = 0.0
            start_time = time.time()
            for i, (sentences, tags) in enumerate(train_loader):
                sentences, tags = sentences.to(device), tags.to(device)
                model.zero_grad()
                tag_scores = model(sentences)
                loss = loss_function(tag_scores.view(-1, len(tag_to_ix)), tags.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                # Log and print dev-set accuracy, time taken, and estimated time left every 500 sentences
                if (i + 1) % 500 == 0:
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for dev_sentences, dev_tags in dev_loader:
                            dev_sentences, dev_tags = dev_sentences.to(device), dev_tags.to(device)
                            dev_tag_scores = model(dev_sentences)
                            predicted_tags = torch.argmax(dev_tag_scores, dim=2)
                            for pred, true in zip(predicted_tags.view(-1), dev_tags.view(-1)):
                                if "ner" in train_path and pred.item() == tag_to_ix["O"] and true.item() == tag_to_ix["O"]:
                                    continue  # Skip cases where both are 'O' in NER
                                if pred.item() == true.item():
                                    correct += 1
                                total += 1
                    accuracy = correct / total if total > 0 else 0

                    elapsed_time = time.time() - start_time
                    sentences_processed = (i + 1) * BATCH_SIZE
                    time_per_500 = elapsed_time / (sentences_processed / 500)
                    remaining_batches = (len(train_data) - sentences_processed) / 500
                    estimated_time_left = remaining_batches * time_per_500

                    log_f.write(f"After {sentences_processed} sentences: Dev Accuracy = {accuracy:.4f}, Time = {time_per_500:.2f}s, Estimated Time Left = {estimated_time_left:.2f}s\n")
                    print(f"After {sentences_processed} sentences: Dev Accuracy = {accuracy:.4f}, Time = {time_per_500:.2f}s, Estimated Time Left = {estimated_time_left:.2f}s")

            log_f.write(f"Epoch {epoch+1}: Loss = {total_loss:.4f}\n")
            print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

    return model

# === Entry Point ===
if __name__ == "__main__":
    import sys
    repr = sys.argv[1]  # Currently unused
    trainFile = sys.argv[2]  # Path to the training file
    modelFile = sys.argv[3]  # Path to save the trained model

    # Derive devFile from trainFile by replacing 'train' with 'dev'
    devFile = trainFile.replace("train", "dev")

    # Train the model
    model = train_model(trainFile, devFile)

    # Save the trained model
    torch.save(model.state_dict(), modelFile)
