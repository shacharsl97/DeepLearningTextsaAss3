import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import numpy as np
import time
from torch.nn.utils.rnn import pad_sequence
import pickle

# === Hyperparameters ===
EMBEDDING_DIM = 200
CHAR_EMBEDDING_DIM = 100
HIDDEN_DIM = 128
EPOCHS = 5
LEARNING_RATE = 0.015
BATCH_SIZE = 25
DROPOUT = 0.25

PAD_CHAR_IDX = 0

GPU_NUMBER = 0

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

class CharTaggingDataset(Dataset):
    def __init__(self, data_path, char_to_ix, tag_to_ix):
        self.sentences = []
        self.labels = []
        self.char_to_ix = char_to_ix
        self.tag_to_ix = tag_to_ix
        with open(data_path) as f:
            sentence = []
            tags = []
            for line in f:
                line = line.strip()
                if line == "":
                    if sentence:
                        self.sentences.append([torch.tensor(word) for word in sentence])
                        self.labels.append(torch.tensor(tags))
                        sentence = []
                        tags = []
                else:
                    word, tag = line.split()
                    char_indices = [char_to_ix.get(char, char_to_ix["<UNK>"]) for char in word]
                    sentence.append(char_indices)
                    tags.append(tag_to_ix[tag])
            if sentence:
                self.sentences.append([torch.tensor(word) for word in sentence])
                self.labels.append(torch.tensor(tags))

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]

    # Collate function for DataLoader
    @staticmethod
    def collate_fn(batch):
        sentences, tags = zip(*batch)
        # Find max sentence length and max word length in the batch
        max_sent_len = max(len(s) for s in sentences)
        max_word_len = max((len(w) for s in sentences for w in s), default=1)
        # Pad each word to max_word_len, then pad each sentence to max_sent_len
        padded_sentences = []
        for s in sentences:
            padded_words = [torch.cat([w, torch.full((max_word_len - len(w),), PAD_CHAR_IDX, dtype=torch.long)]) if len(w) < max_word_len else w[:max_word_len] for w in s]
            # Pad sentence with PAD_CHAR_IDX words if needed
            if len(padded_words) < max_sent_len:
                pad_word = torch.full((max_word_len,), PAD_CHAR_IDX, dtype=torch.long)
                padded_words += [pad_word] * (max_sent_len - len(padded_words))
            padded_sentences.append(torch.stack(padded_words))
        sentences_padded = torch.stack(padded_sentences)  # (batch, max_sent_len, max_word_len)
        # Pad tags
        tags_padded = [torch.cat([t, torch.full((max_sent_len - len(t),), -1, dtype=torch.long)]) if len(t) < max_sent_len else t[:max_sent_len] for t in tags]
        tags_padded = torch.stack(tags_padded)
        return sentences_padded, tags_padded

class PrefixSuffixTaggingDataset(Dataset):
    def __init__(self, data_path, word_to_ix, prefix_to_ix, suffix_to_ix, tag_to_ix):
        self.sentences = []
        self.prefixes = []
        self.suffixes = []
        self.labels = []
        self.word_to_ix = word_to_ix
        self.prefix_to_ix = prefix_to_ix
        self.suffix_to_ix = suffix_to_ix
        self.tag_to_ix = tag_to_ix
        with open(data_path) as f:
            sentence = []
            prefixes = []
            suffixes = []
            tags = []
            for line in f:
                line = line.strip()
                if line == "":
                    if sentence:
                        self.sentences.append(torch.tensor(sentence))
                        self.prefixes.append(torch.tensor(prefixes))
                        self.suffixes.append(torch.tensor(suffixes))
                        self.labels.append(torch.tensor(tags))
                        sentence = []
                        prefixes = []
                        suffixes = []
                        tags = []
                else:
                    word, tag = line.split()
                    word_idx = word_to_ix.get(word, word_to_ix["<UNK>"])
                    pad_word = word + ("<PAD>" * (3 - len(word))) if len(word) < 3 else word
                    prefix = pad_word[:3]
                    suffix = pad_word[-3:]
                    prefix_idx = prefix_to_ix.get(prefix, prefix_to_ix["<PAD>"])
                    suffix_idx = suffix_to_ix.get(suffix, suffix_to_ix["<PAD>"])
                    sentence.append(word_idx)
                    prefixes.append(prefix_idx)
                    suffixes.append(suffix_idx)
                    tags.append(tag_to_ix[tag])
            if sentence:
                self.sentences.append(torch.tensor(sentence))
                self.prefixes.append(torch.tensor(prefixes))
                self.suffixes.append(torch.tensor(suffixes))
                self.labels.append(torch.tensor(tags))

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.prefixes[idx], self.suffixes[idx], self.labels[idx]

    @staticmethod
    def collate_fn(batch):
        sentences, prefixes, suffixes, tags = zip(*batch)
        sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)
        prefixes_padded = pad_sequence(prefixes, batch_first=True, padding_value=0)
        suffixes_padded = pad_sequence(suffixes, batch_first=True, padding_value=0)
        tags_padded = pad_sequence(tags, batch_first=True, padding_value=-1)
        return sentences_padded, prefixes_padded, suffixes_padded, tags_padded

class WordCharTaggingDataset(Dataset):
    """
    For mode 'd': returns (word_indices, char_indices, tags) for each sentence.
    """
    def __init__(self, data_path, word_to_ix, char_to_ix, tag_to_ix):
        self.word_sentences = []
        self.char_sentences = []
        self.labels = []
        with open(data_path) as f:
            word_sentence = []
            char_sentence = []
            tags = []
            for line in f:
                line = line.strip()
                if line == "":
                    if word_sentence:
                        self.word_sentences.append(torch.tensor(word_sentence))
                        self.char_sentences.append([torch.tensor(w) for w in char_sentence])
                        self.labels.append(torch.tensor(tags))
                        word_sentence = []
                        char_sentence = []
                        tags = []
                else:
                    word, tag = line.split()
                    word_idx = word_to_ix.get(word, word_to_ix["<UNK>"])
                    char_indices = [char_to_ix.get(char, char_to_ix["<UNK>"]) for char in word]
                    word_sentence.append(word_idx)
                    char_sentence.append(char_indices)
                    tags.append(tag_to_ix[tag])
            if word_sentence:
                self.word_sentences.append(torch.tensor(word_sentence))
                self.char_sentences.append([torch.tensor(w) for w in char_sentence])
                self.labels.append(torch.tensor(tags))

    def __len__(self):
        return len(self.word_sentences)

    def __getitem__(self, idx):
        return self.word_sentences[idx], self.char_sentences[idx], self.labels[idx]

    @staticmethod
    def collate_fn(batch):
        word_sentences, char_sentences, tags = zip(*batch)
        # Pad word sentences
        word_sentences_padded = pad_sequence(word_sentences, batch_first=True, padding_value=0)
        # Pad char sentences
        max_sent_len = max(len(s) for s in char_sentences)
        max_word_len = max((len(w) for s in char_sentences for w in s), default=1)
        padded_char_sentences = []
        for s in char_sentences:
            padded_words = [torch.cat([w, torch.full((max_word_len - len(w),), PAD_CHAR_IDX, dtype=torch.long)]) if len(w) < max_word_len else w[:max_word_len] for w in s]
            if len(padded_words) < max_sent_len:
                pad_word = torch.full((max_word_len,), PAD_CHAR_IDX, dtype=torch.long)
                padded_words += [pad_word] * (max_sent_len - len(padded_words))
            padded_char_sentences.append(torch.stack(padded_words))
        char_sentences_padded = torch.stack(padded_char_sentences)  # (batch, max_sent_len, max_word_len)
        # Pad tags
        tags_padded = [torch.cat([t, torch.full((max_sent_len - len(t),), -1, dtype=torch.long)]) if len(t) < max_sent_len else t[:max_sent_len] for t in tags]
        tags_padded = torch.stack(tags_padded)
        return word_sentences_padded, char_sentences_padded, tags_padded

# === Model Definition ===
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, char_vocab_size, prefix_vocab_size, suffix_vocab_size,
                 tagset_size, embedding_dim, char_embedding_dim, hidden_dim, mode):
        super(BiLSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.char_embedding_dim = char_embedding_dim
        self.dropout = nn.Dropout(DROPOUT)
        self.lstm_fwd = nn.LSTMCell(embedding_dim, hidden_dim)
        self.lstm_bwd = nn.LSTMCell(embedding_dim, hidden_dim)
        self.lstm2_fwd = nn.LSTMCell(hidden_dim * 2, hidden_dim)
        self.lstm2_bwd = nn.LSTMCell(hidden_dim * 2, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)
        self.mode = mode

        if mode in ['a', 'c', 'd']:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if mode in ['b', 'd']:
            self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim, padding_idx=PAD_CHAR_IDX)
            self.char_lstm_fwd = nn.LSTMCell(char_embedding_dim, embedding_dim // 2)
            self.char_lstm_bwd = nn.LSTMCell(char_embedding_dim, embedding_dim // 2)
        if mode == 'c':
            self.prefix_embedding = nn.Embedding(prefix_vocab_size, embedding_dim)
            self.suffix_embedding = nn.Embedding(suffix_vocab_size, embedding_dim)
        if mode == 'd':
            self.embedding_final_linear = nn.Linear(embedding_dim + embedding_dim, embedding_dim)

    def forward(self, sentence, char_sentence=None, prefix=None, suffix=None):
        if self.mode == 'a':
            embeds = self.embedding(sentence)  # Shape: (batch_size, seq_len, embedding_dim)
        elif self.mode == 'b':
            batch_size, seq_len, max_word_len = char_sentence.size()
            char_embeds = self.char_embedding(char_sentence)  # (batch, seq_len, max_word_len, char_embedding_dim)
            word_embeds = []
            for i in range(seq_len):
                # Forward direction
                h_fwd = torch.zeros(batch_size, self.embedding_dim // 2, device=char_embeds.device)
                c_fwd = torch.zeros(batch_size, self.embedding_dim // 2, device=char_embeds.device)
                # Backward direction
                h_bwd = torch.zeros(batch_size, self.embedding_dim // 2, device=char_embeds.device)
                c_bwd = torch.zeros(batch_size, self.embedding_dim // 2, device=char_embeds.device)
                # Forward pass
                for j in range(max_word_len):
                    h_fwd, c_fwd = self.char_lstm_fwd(char_embeds[:, i, j, :], (h_fwd, c_fwd))
                # Backward pass
                for j in reversed(range(max_word_len)):
                    h_bwd, c_bwd = self.char_lstm_bwd(char_embeds[:, i, j, :], (h_bwd, c_bwd))
                # Concatenate final states
                word_embeds.append(torch.cat([h_fwd, h_bwd], dim=1))
            embeds = torch.stack(word_embeds, dim=1)  # (batch, seq_len, embedding_dim)
        elif self.mode == 'c':
            word_embeds = self.embedding(sentence)  # (batch, seq_len, embedding_dim)
            prefix_embeds = self.prefix_embedding(prefix)  # (batch, seq_len, embedding_dim)
            suffix_embeds = self.suffix_embedding(suffix)  # (batch, seq_len, embedding_dim)
            embeds = word_embeds + prefix_embeds + suffix_embeds
        elif self.mode == 'd':
            # sentence: (batch, seq_len), char_sentence: (batch, seq_len, max_word_len)
            word_embeds = self.embedding(sentence)  # (batch, seq_len, embedding_dim)
            batch_size, seq_len, max_word_len = char_sentence.size()
            char_embeds = self.char_embedding(char_sentence)  # (batch, seq_len, max_word_len, char_embedding_dim)
            char_word_embeds = []
            for i in range(seq_len):
                # Forward direction
                h_fwd = torch.zeros(batch_size, self.embedding_dim // 2, device=char_embeds.device)
                c_fwd = torch.zeros(batch_size, self.embedding_dim // 2, device=char_embeds.device)
                # Backward direction
                h_bwd = torch.zeros(batch_size, self.embedding_dim // 2, device=char_embeds.device)
                c_bwd = torch.zeros(batch_size, self.embedding_dim // 2, device=char_embeds.device)
                # Forward pass
                for j in range(max_word_len):
                    h_fwd, c_fwd = self.char_lstm_fwd(char_embeds[:, i, j, :], (h_fwd, c_fwd))
                # Backward pass
                for j in reversed(range(max_word_len)):
                    h_bwd, c_bwd = self.char_lstm_bwd(char_embeds[:, i, j, :], (h_bwd, c_bwd))
                char_word_embeds.append(torch.cat([h_fwd, h_bwd], dim=1))
            char_word_embeds = torch.stack(char_word_embeds, dim=1)  # (batch, seq_len, embedding_dim)
            concat_embeds = torch.cat([word_embeds, char_word_embeds], dim=2)  # (batch, seq_len, embedding_dim*2)
            embeds = self.embedding_final_linear(concat_embeds)  # (batch, seq_len, embedding_dim)

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

def build_char_vocab(data_path):
    char_to_ix = {"<UNK>": 0}
    tag_to_ix = {}
    with open(data_path) as f:
        for line in f:
            if line.strip() == "":
                continue
            word, tag = line.strip().split()
            for char in word:
                if char not in char_to_ix:
                    char_to_ix[char] = len(char_to_ix)
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)
    return char_to_ix, tag_to_ix

def build_prefix_suffix_vocab(data_path):
    """
    Builds prefix, suffix, word, and tag vocabularies from all words in the dataset.
    Prefix/suffix is 3 chars, padded with '<PAD>' if word is shorter.
    Returns: word_to_ix, prefix_to_ix, suffix_to_ix, tag_to_ix
    """
    word_to_ix = {"<UNK>": 0}
    prefix_to_ix = {"<PAD>": 0}
    suffix_to_ix = {"<PAD>": 0}
    tag_to_ix = {}
    with open(data_path) as f:
        for line in f:
            if line.strip() == "":
                continue
            word, tag = line.strip().split()
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
            # Pad word if needed
            pad_word = word + ("<PAD>" * (3 - len(word))) if len(word) < 3 else word
            prefix = pad_word[:3]
            suffix = pad_word[-3:]
            if prefix not in prefix_to_ix:
                prefix_to_ix[prefix] = len(prefix_to_ix)
            if suffix not in suffix_to_ix:
                suffix_to_ix[suffix] = len(suffix_to_ix)
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)
    return word_to_ix, prefix_to_ix, suffix_to_ix, tag_to_ix

# === Training Procedure ===
def train_model(train_path, dev_path, mode):
    device = torch.device(f"cuda:{GPU_NUMBER}" if torch.cuda.is_available() else "cpu")
    vocab_size = 0
    char_vocab_size = 0
    prefix_vocab_size = 0
    suffix_vocab_size = 0
    if mode == 'a':
        word_to_ix, tag_to_ix = build_vocab(train_path)
        train_data = TaggingDataset(train_path, word_to_ix, tag_to_ix)
        dev_data = TaggingDataset(dev_path, word_to_ix, tag_to_ix)
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=TaggingDataset.collate_fn)
        dev_loader = DataLoader(dev_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=TaggingDataset.collate_fn)
        vocab_size = len(word_to_ix)
        model = BiLSTMTagger(
            vocab_size, char_vocab_size, prefix_vocab_size, suffix_vocab_size,
            len(tag_to_ix), EMBEDDING_DIM, CHAR_EMBEDDING_DIM, HIDDEN_DIM, mode
        ).to(device)

    if mode == 'b':
        char_to_ix, tag_to_ix = build_char_vocab(train_path)
        train_data = CharTaggingDataset(train_path, char_to_ix, tag_to_ix)
        dev_data = CharTaggingDataset(dev_path, char_to_ix, tag_to_ix)
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=CharTaggingDataset.collate_fn)
        dev_loader = DataLoader(dev_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=CharTaggingDataset.collate_fn)
        char_vocab_size = len(char_to_ix)
        model = BiLSTMTagger(
            vocab_size, char_vocab_size, prefix_vocab_size, suffix_vocab_size,
            len(tag_to_ix), EMBEDDING_DIM, CHAR_EMBEDDING_DIM, HIDDEN_DIM, mode
        ).to(device)

    if mode == 'c':
        word_to_ix, prefix_to_ix, suffix_to_ix, tag_to_ix = build_prefix_suffix_vocab(train_path)
        train_data = PrefixSuffixTaggingDataset(train_path, word_to_ix, prefix_to_ix, suffix_to_ix, tag_to_ix)
        dev_data = PrefixSuffixTaggingDataset(dev_path, word_to_ix, prefix_to_ix, suffix_to_ix, tag_to_ix)
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=PrefixSuffixTaggingDataset.collate_fn)
        dev_loader = DataLoader(dev_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=PrefixSuffixTaggingDataset.collate_fn)
        vocab_size = len(word_to_ix)
        prefix_vocab_size = len(prefix_to_ix)
        suffix_vocab_size = len(suffix_to_ix)
        model = BiLSTMTagger(
            vocab_size, char_vocab_size, prefix_vocab_size, suffix_vocab_size,
            len(tag_to_ix), EMBEDDING_DIM, CHAR_EMBEDDING_DIM, HIDDEN_DIM, mode
        ).to(device)

    if mode == 'd':
        word_to_ix, tag_to_ix = build_vocab(train_path)
        char_to_ix, _ = build_char_vocab(train_path)
        train_data = WordCharTaggingDataset(train_path, word_to_ix, char_to_ix, tag_to_ix)
        dev_data = WordCharTaggingDataset(dev_path, word_to_ix, char_to_ix, tag_to_ix)
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=WordCharTaggingDataset.collate_fn)
        dev_loader = DataLoader(dev_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=WordCharTaggingDataset.collate_fn)
        vocab_size = len(word_to_ix)
        char_vocab_size = len(char_to_ix)
        model = BiLSTMTagger(
            vocab_size, char_vocab_size, prefix_vocab_size, suffix_vocab_size,
            len(tag_to_ix), EMBEDDING_DIM, CHAR_EMBEDDING_DIM, HIDDEN_DIM, mode
        ).to(device)

    
    log_file_name = f"log_{mode}_{train_path[0:3]}.txt"
    loss_function = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.65, patience=2)

    print(f"Training on {len(train_data)} sentences\n Starting training...")

    with open(log_file_name, "w") as log_f:
        for epoch in range(EPOCHS):
            total_loss = 0.0
            start_time = time.time()
            for i, batch in enumerate(train_loader):
                if mode == 'a':
                    sentences, tags = batch
                    sentences, tags = sentences.to(device), tags.to(device)
                    model.zero_grad()
                    tag_scores = model(sentences)
                elif mode == 'b':
                    sentences, tags = batch
                    sentences, tags = sentences.to(device), tags.to(device)
                    model.zero_grad()
                    tag_scores = model(None, sentences)
                elif mode == 'c':
                    sentences, prefixes, suffixes, tags = batch
                    sentences, prefixes, suffixes, tags = sentences.to(device), prefixes.to(device), suffixes.to(device), tags.to(device)
                    model.zero_grad()
                    tag_scores = model(sentences, None, prefixes, suffixes)
                elif mode == 'd':
                    word_sentences, char_sentences, tags = batch
                    word_sentences, char_sentences, tags = word_sentences.to(device), char_sentences.to(device), tags.to(device)
                    model.zero_grad()
                    tag_scores = model(word_sentences, char_sentences)
                loss = loss_function(tag_scores.view(-1, tag_scores.size(-1)), tags.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                # Log and print dev-set accuracy, time taken, and estimated time left every 500 sentences
                if (i + 1) % (500 / BATCH_SIZE) == 0:
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for dev_batch in dev_loader:
                            if mode == 'a':
                                dev_sentences, dev_tags = dev_batch
                                dev_sentences, dev_tags = dev_sentences.to(device), dev_tags.to(device)
                                dev_tag_scores = model(dev_sentences)
                            elif mode == 'b':
                                dev_sentences, dev_tags = dev_batch
                                dev_sentences, dev_tags = dev_sentences.to(device), dev_tags.to(device)
                                dev_tag_scores = model(None, dev_sentences)
                            elif mode == 'c':
                                dev_sentences, dev_prefixes, dev_suffixes, dev_tags = dev_batch
                                dev_sentences, dev_prefixes, dev_suffixes, dev_tags = dev_sentences.to(device), dev_prefixes.to(device), dev_suffixes.to(device), dev_tags.to(device)
                                dev_tag_scores = model(dev_sentences, None, dev_prefixes, dev_suffixes)
                            elif mode == 'd':
                                dev_word_sentences, dev_char_sentences, dev_tags = dev_batch
                                dev_word_sentences, dev_char_sentences, dev_tags = dev_word_sentences.to(device), dev_char_sentences.to(device), dev_tags.to(device)
                                dev_tag_scores = model(dev_word_sentences, dev_char_sentences)
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
                        for batch_idx, train_batch in enumerate(train_loader):
                            if batch_idx >= sample_size:
                                break
                            if mode == 'a':
                                train_sentences, train_tags = train_batch
                                train_sentences, train_tags = train_sentences.to(device), train_tags.to(device)
                                train_tag_scores = model(train_sentences)
                            elif mode == 'b':
                                train_sentences, train_tags = train_batch
                                train_sentences, train_tags = train_sentences.to(device), train_tags.to(device)
                                train_tag_scores = model(None, train_sentences)
                            elif mode == 'c':
                                train_sentences, train_prefixes, train_suffixes, train_tags = train_batch
                                train_sentences, train_prefixes, train_suffixes, train_tags = train_sentences.to(device), train_prefixes.to(device), train_suffixes.to(device), train_tags.to(device)
                                train_tag_scores = model(train_sentences, None, train_prefixes, train_suffixes)
                            elif mode == 'd':
                                train_word_sentences, train_char_sentences, train_tags = train_batch
                                train_word_sentences, train_char_sentences, train_tags = train_word_sentences.to(device), train_char_sentences.to(device), train_tags.to(device)
                                train_tag_scores = model(train_word_sentences, train_char_sentences)
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
                for train_batch in train_loader:
                    if mode == 'a':
                        train_sentences, train_tags = train_batch
                        train_sentences, train_tags = train_sentences.to(device), train_tags.to(device)
                        train_tag_scores = model(train_sentences)
                    elif mode == 'b':
                        train_sentences, train_tags = train_batch
                        train_sentences, train_tags = train_sentences.to(device), train_tags.to(device)
                        train_tag_scores = model(None, train_sentences)
                    elif mode == 'c':
                        train_sentences, train_prefixes, train_suffixes, train_tags = train_batch
                        train_sentences, train_prefixes, train_suffixes, train_tags = train_sentences.to(device), train_prefixes.to(device), train_suffixes.to(device), train_tags.to(device)
                        train_tag_scores = model(train_sentences, None, train_prefixes, train_suffixes)
                    elif mode == 'd':
                        train_word_sentences, train_char_sentences, train_tags = train_batch
                        train_word_sentences, train_char_sentences, train_tags = train_word_sentences.to(device), train_char_sentences.to(device), train_tags.to(device)
                        train_tag_scores = model(train_word_sentences, train_char_sentences)
                    train_loss += loss_function(train_tag_scores.view(-1, train_tag_scores.size(-1)), train_tags.view(-1)).item()

            log_f.write(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Dev Loss = {total_loss:.4f}\n")
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Dev Loss = {total_loss:.4f}")

            log_f.write(f"Epoch {epoch+1}: Loss = {total_loss:.4f}\n")
            print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

    if mode == 'a':
        return model, (word_to_ix, tag_to_ix)
    elif mode == 'b':
        return model, (char_to_ix, tag_to_ix)
    elif mode == 'c':
        return model, (word_to_ix, prefix_to_ix, suffix_to_ix, tag_to_ix)
    elif mode == 'd':
        return model, (word_to_ix, char_to_ix, tag_to_ix)

# === Entry Point ===
if __name__ == "__main__":
    import sys
    mode = sys.argv[1]  # Currently unused
    trainFile = sys.argv[2]  # Path to the training file
    modelFile = sys.argv[3]  # Path to save the trained model

    if mode not in ['a', 'b', 'c', 'd']:
        print("Invalid mode. Use 'a', 'b', 'c', or 'd'. [1st argument]")
        sys.exit(1)

    if not os.path.exists(trainFile):
        print(f"Training file {trainFile} does not exist. [2nd argument]")
        sys.exit(1)

    # Derive devFile from trainFile by replacing 'train' with 'dev'
    devFile = trainFile.replace("train", "dev")

    # Train the model
    model, vocabs = train_model(trainFile, devFile, mode)

    # Save the trained model
    torch.save(model.state_dict(), modelFile)

    # Save vocabularies and mode for prediction
    vocabs_to_save = {'mode': mode}
    if mode == 'a':
        word_to_ix, tag_to_ix = vocabs
        vocabs_to_save['word_to_ix'] = word_to_ix
        vocabs_to_save['tag_to_ix'] = tag_to_ix
    elif mode == 'b':
        char_to_ix, tag_to_ix = vocabs
        vocabs_to_save['char_to_ix'] = char_to_ix
        vocabs_to_save['tag_to_ix'] = tag_to_ix
    elif mode == 'c':
        word_to_ix, prefix_to_ix, suffix_to_ix, tag_to_ix = vocabs
        vocabs_to_save['word_to_ix'] = word_to_ix
        vocabs_to_save['prefix_to_ix'] = prefix_to_ix
        vocabs_to_save['suffix_to_ix'] = suffix_to_ix
        vocabs_to_save['tag_to_ix'] = tag_to_ix
    elif mode == 'd':
        word_to_ix, char_to_ix, tag_to_ix = vocabs
        vocabs_to_save['word_to_ix'] = word_to_ix
        vocabs_to_save['char_to_ix'] = char_to_ix
        vocabs_to_save['tag_to_ix'] = tag_to_ix
    with open(modelFile + '.vocabs', 'wb') as f:
        pickle.dump(vocabs_to_save, f)

    print(f"Model saved to {modelFile}")
    print(f"Vocabularies saved to {modelFile}.vocabs")
