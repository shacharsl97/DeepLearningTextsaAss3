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
EMBEDDING_DIM = 200
HIDDEN_DIM = 128
EPOCHS = 5
LEARNING_RATE = 0.015
BATCH_SIZE = 200
DROPOUT = 0.3

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
        self.dropout = nn.Dropout(DROPOUT)
        self.lstm_fwd = nn.LSTMCell(embedding_dim, hidden_dim)
        self.lstm_bwd = nn.LSTMCell(embedding_dim, hidden_dim)
        self.lstm2_fwd = nn.LSTMCell(hidden_dim * 2, hidden_dim)
        self.lstm2_bwd = nn.LSTMCell(hidden_dim * 2, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)
        # self.layer_norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, sentence):
        embeds = self.embedding(sentence)  # Shape: (batch_size, seq_len, embedding_dim)
        batch_size, seq_len, _ = embeds.size()

        # First BiLSTM layer
        h_fwd = torch.zeros(batch_size, self.hidden_dim, device=embeds.device)
        c_fwd = torch.zeros(batch_size, self.hidden_dim, device=embeds.device)
        h_bwd = torch.zeros(batch_size, self.hidden_dim, device=embeds.device)
        c_bwd = torch.zeros(batch_size, self.hidden_dim, device=embeds.device)

        outputs_fwd = []
        for i in range(seq_len):
            h_fwd, c_fwd = self.lstm_fwd(embeds[:, i, :], (h_fwd, c_fwd))  # Process each time step
            outputs_fwd.append(h_fwd)

        outputs_bwd = []
        for i in reversed(range(seq_len)):
            h_bwd, c_bwd = self.lstm_bwd(embeds[:, i, :], (h_bwd, c_bwd))  # Process each time step in reverse
            outputs_bwd.insert(0, h_bwd)

        outputs = [torch.cat((f, b), dim=1) for f, b in zip(outputs_fwd, outputs_bwd)]
        outputs = torch.stack(outputs, dim=1)  # Shape: (batch_size, seq_len, hidden_dim * 2)

        # Second BiLSTM layer
        h2_fwd = torch.zeros(batch_size, self.hidden_dim, device=embeds.device)
        c2_fwd = torch.zeros(batch_size, self.hidden_dim, device=embeds.device)
        h2_bwd = torch.zeros(batch_size, self.hidden_dim, device=embeds.device)
        c2_bwd = torch.zeros(batch_size, self.hidden_dim, device=embeds.device)

        outputs2_fwd = []
        for i in range(seq_len):
            h2_fwd, c2_fwd = self.lstm2_fwd(outputs[:, i, :], (h2_fwd, c2_fwd))  # Process each time step
            outputs2_fwd.append(h2_fwd)

        outputs2_bwd = []
        for i in reversed(range(seq_len)):
            h2_bwd, c2_bwd = self.lstm2_bwd(outputs[:, i, :], (h2_bwd, c2_bwd))  # Process each time step in reverse
            outputs2_bwd.insert(0, h2_bwd)

        outputs2 = [torch.cat((f, b), dim=1) for f, b in zip(outputs2_fwd, outputs2_bwd)]
        outputs2 = torch.stack(outputs2, dim=1)  # Shape: (batch_size, seq_len, hidden_dim * 2)
        outputs2 = self.dropout(outputs2)  # Apply dropout
        # outputs2 = self.layer_norm(outputs2)     # Apply layer normalization
        tag_space = self.hidden2tag(outputs2)  # Shape: (batch_size, seq_len, tagset_size)
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
def train_model(train_path, dev_path, mode):
    device = torch.device(f"cuda:{GPU_NUMBER}" if torch.cuda.is_available() else "cpu")

    word_to_ix, tag_to_ix = build_vocab(train_path)
    train_data = TaggingDataset(train_path, word_to_ix, tag_to_ix)
    dev_data = TaggingDataset(dev_path, word_to_ix, tag_to_ix)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=TaggingDataset.collate_fn)
    dev_loader = DataLoader(dev_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=TaggingDataset.collate_fn)

    log_file = "train_log_pos.txt" if "pos" in train_path else "train_log_ner.txt"

    model = BiLSTMTagger(len(word_to_ix), len(tag_to_ix), EMBEDDING_DIM, HIDDEN_DIM).to(device)
    loss_function = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)


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
                if (i + 1) % (500 / BATCH_SIZE) == 0:
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for dev_sentences, dev_tags in dev_loader:
                            dev_sentences, dev_tags = dev_sentences.to(device), dev_tags.to(device)
                            dev_tag_scores = model(dev_sentences)
                            predicted_tags = torch.argmax(dev_tag_scores, dim=2)
                            mask = dev_tags != -1
                            if "ner" in train_path:
                                ner_mask = ~((predicted_tags == tag_to_ix["O"]) & (dev_tags == tag_to_ix["O"]))
                                mask &= ner_mask
                            correct += (predicted_tags[mask] == dev_tags[mask]).sum().item()
                            total += mask.sum().item()
                    accuracy = correct / total if total > 0 else 0

                    correct_train = 0
                    total_train = 0
                    sample_size = min(len(train_loader), 10)
                    with torch.no_grad():
                        for batch_idx, (train_sentences, train_tags) in enumerate(train_loader):
                            if batch_idx >= sample_size:
                                break
                            train_sentences, train_tags = train_sentences.to(device), train_tags.to(device)
                            train_tag_scores = model(train_sentences)
                            predicted_train_tags = torch.argmax(train_tag_scores, dim=2)
                            mask_train = train_tags != -1
                            if "ner" in train_path:
                                ner_mask = ~((predicted_train_tags == tag_to_ix["O"]) & (train_tags == tag_to_ix["O"]))
                                mask_train &= ner_mask
                            correct_train += (predicted_train_tags[mask_train] == train_tags[mask_train]).sum().item()
                            total_train += mask_train.sum().item()
                    train_accuracy = correct_train / total_train if total_train > 0 else 0


                    elapsed_time = time.time() - start_time
                    sentences_processed = (i + 1) * BATCH_SIZE
                    time_per_500 = elapsed_time / (sentences_processed / 500)
                    remaining_batches = (len(train_data) - sentences_processed) / 500
                    estimated_time_left = remaining_batches * time_per_500

                    log_f.write(f"After {sentences_processed} sentences: Train Accuracy = {train_accuracy:.4f}, Dev Accuracy = {accuracy:.4f}, Time = {time_per_500:.2f}s, Estimated Time Left = {estimated_time_left:.2f}s\n")
                    print(f"After {sentences_processed} sentences: Train Accuracy = {train_accuracy:.4f}, Dev Accuracy = {accuracy:.4f}, Time = {time_per_500:.2f}s, Estimated Time Left = {estimated_time_left:.2f}s")

                    scheduler.step(accuracy)

            # Calculate train loss at the end of the epoch
            train_loss = 0.0
            with torch.no_grad():
                for train_sentences, train_tags in train_loader:
                    train_sentences, train_tags = train_sentences.to(device), train_tags.to(device)
                    train_tag_scores = model(train_sentences)
                    train_loss += loss_function(train_tag_scores.view(-1, len(tag_to_ix)), train_tags.view(-1)).item()

            log_f.write(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Dev Loss = {total_loss:.4f}\n")
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Dev Loss = {total_loss:.4f}")

            log_f.write(f"Epoch {epoch+1}: Loss = {total_loss:.4f}\n")
            print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

    return model

# === Entry Point ===
if __name__ == "__main__":
    import sys
    mode = sys.argv[1]  # Currently unused
    trainFile = sys.argv[2]  # Path to the training file
    modelFile = sys.argv[3]  # Path to save the trained model

    # Derive devFile from trainFile by replacing 'train' with 'dev'
    devFile = trainFile.replace("train", "dev")

    # Train the model
    model = train_model(trainFile, devFile, mode)

    # Save the trained model
    torch.save(model.state_dict(), modelFile)
