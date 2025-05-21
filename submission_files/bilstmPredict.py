import sys
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from bilstmTrain import (
    TaggingDataset, CharTaggingDataset, PrefixSuffixTaggingDataset, WordCharTaggingDataset,
    BiLSTMTagger, build_vocab, build_char_vocab, build_prefix_suffix_vocab,
    EMBEDDING_DIM, CHAR_EMBEDDING_DIM, HIDDEN_DIM, BATCH_SIZE, PAD_CHAR_IDX
)

GPU_NUMBER = 0

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 bilstmPredict.py <mode> <modelFile> <inputFile>")
        sys.exit(1)

    mode = sys.argv[1]
    modelFile = sys.argv[2]
    inputFile = sys.argv[3]

    if 'pos' in modelFile:
        outputFile = 'test4.pos'
    elif 'ner' in modelFile:
        outputFile = 'test4.ner'
    else:
        print("Model file name must contain 'pos' or 'ner'.")
        sys.exit(1)

    device = torch.device(f'cuda:{GPU_NUMBER}' if torch.cuda.is_available() else 'cpu')

    print(f"Mode: {mode}")
    print(f"Model file: {modelFile}")
    print(f"Input file: {inputFile}")
    print(f"Output file: {outputFile}")
    print(f"Device: {device}")

    # Read input file as sentences
    with open(inputFile) as f:
        input_lines = [line.strip() for line in f]
    print(f"Loaded {len(input_lines)} lines from input file.")
    # Build sentences (list of list of words)
    sentences = []
    current = []
    for line in input_lines:
        if line.strip() == '':
            if current:
                sentences.append(current)
                current = []
        else:
            current.append(line.strip())
    if current:
        sentences.append(current)
    print(f"Parsed {len(sentences)} sentences from input.")

    # Load vocabularies and mode from pickle file
    import pickle
    with open(modelFile + '.vocabs', 'rb') as f:
        vocabs = pickle.load(f)
    print(f"Loaded vocabs: {list(vocabs.keys())}")
    if mode != vocabs['mode']:
        print(f"Model mode {vocabs['mode']} does not match input mode {mode}.")
        sys.exit(1)
        
    tag_to_ix = vocabs['tag_to_ix']
    ix_to_tag = {v: k for k, v in tag_to_ix.items()}
    word_to_ix = vocabs.get('word_to_ix', None)
    char_to_ix = vocabs.get('char_to_ix', None)
    prefix_to_ix = vocabs.get('prefix_to_ix', None)
    suffix_to_ix = vocabs.get('suffix_to_ix', None)

    # Load model with correct vocab sizes
    if mode == 'a':
        vocab_size = len(word_to_ix)
        char_vocab_size = 0
        prefix_vocab_size = 0
        suffix_vocab_size = 0
    elif mode == 'b':
        vocab_size = 0
        char_vocab_size = len(char_to_ix)
        prefix_vocab_size = 0
        suffix_vocab_size = 0
    elif mode == 'c':
        vocab_size = len(word_to_ix)
        char_vocab_size = 0
        prefix_vocab_size = len(prefix_to_ix)
        suffix_vocab_size = len(suffix_to_ix)
    elif mode == 'd':
        vocab_size = len(word_to_ix)
        char_vocab_size = len(char_to_ix)
        prefix_vocab_size = 0
        suffix_vocab_size = 0
    else:
        print("Invalid mode. Please use 'a', 'b', 'c', or 'd'.")
        sys.exit(1)

    model = BiLSTMTagger(
        vocab_size, char_vocab_size, prefix_vocab_size, suffix_vocab_size,
        len(tag_to_ix), EMBEDDING_DIM, CHAR_EMBEDDING_DIM, HIDDEN_DIM, mode
    ).to(device)
    model.load_state_dict(torch.load(modelFile, map_location=device))
    model.eval()
    print("Model loaded and set to eval mode.")

    output_lines = []
    with torch.no_grad():
        for sent in tqdm(sentences):
            if mode == 'a':
                word_idxs = [word_to_ix.get(w, word_to_ix['<UNK>']) for w in sent]
                input_tensor = torch.tensor([word_idxs], dtype=torch.long, device=device)
                tag_scores = model(input_tensor)
            elif mode == 'b':
                char_idxs = [[char_to_ix.get(c, char_to_ix['<UNK>']) for c in w] for w in sent]
                max_word_len = max(len(w) for w in char_idxs)
                padded = [w + [PAD_CHAR_IDX] * (max_word_len - len(w)) for w in char_idxs]
                input_tensor = torch.tensor([[p for p in padded]], dtype=torch.long, device=device).squeeze(0)
                input_tensor = input_tensor.unsqueeze(0)  # (1, seq_len, max_word_len)
                tag_scores = model(None, input_tensor)
            elif mode == 'c':
                word_idxs = [word_to_ix.get(w, word_to_ix['<UNK>']) for w in sent]
                prefix_idxs = []
                suffix_idxs = []
                for w in sent:
                    pad_word = w + ("<PAD>" * (3 - len(w))) if len(w) < 3 else w
                    prefix = pad_word[:3]
                    suffix = pad_word[-3:]
                    prefix_idxs.append(prefix_to_ix.get(prefix, prefix_to_ix['<PAD>']))
                    suffix_idxs.append(suffix_to_ix.get(suffix, suffix_to_ix['<PAD>']))
                input_tensor = torch.tensor([word_idxs], dtype=torch.long, device=device)
                prefix_tensor = torch.tensor([prefix_idxs], dtype=torch.long, device=device)
                suffix_tensor = torch.tensor([suffix_idxs], dtype=torch.long, device=device)
                tag_scores = model(input_tensor, None, prefix_tensor, suffix_tensor)
            elif mode == 'd':
                word_idxs = [word_to_ix.get(w, word_to_ix['<UNK>']) for w in sent]
                char_idxs = [[char_to_ix.get(c, char_to_ix['<UNK>']) for c in w] for w in sent]
                max_word_len = max(len(w) for w in char_idxs)
                padded = [w + [PAD_CHAR_IDX] * (max_word_len - len(w)) for w in char_idxs]
                word_tensor = torch.tensor([word_idxs], dtype=torch.long, device=device)
                char_tensor = torch.tensor([[p for p in padded]], dtype=torch.long, device=device).squeeze(0)
                char_tensor = char_tensor.unsqueeze(0)
                tag_scores = model(word_tensor, char_tensor)
            pred = torch.argmax(tag_scores, dim=2).cpu().numpy()[0]
            for i, w in enumerate(sent):
                tag_str = ix_to_tag.get(pred[i], str(pred[i]))
                output_lines.append(f"{w} {tag_str}")
            output_lines.append("")  # blank line between sentences
    print(f"Writing predictions to {outputFile}")
    with open(outputFile, 'w') as f:
        for line in output_lines:
            f.write(line + '\n')
    print("Done.")

if __name__ == "__main__":
    main()
