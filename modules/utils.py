from modules.vocab import Vocab
from tqdm import tqdm

import torch

def load_embeddings(tokens_vocab: Vocab, embeddings_file: str):
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        # чтобы сильно не мудрить пока, vocab_size и размер эмбеддингов хардкодим для '/content/glove.6B.100d.txt'
        vocab_size = 400000
        embeddings_size = 100
        print('Vocab size: %s\tEmbeddings size: %s' % (vocab_size, embeddings_size))
        embeddings_size = int(embeddings_size)
        embeddings_matrix = torch.rand((len(tokens_vocab), embeddings_size))
        paddings = torch.zeros(embeddings_size)
        embeddings_matrix[0] = paddings
        for line in tqdm(f):
            word, *weights = line.split()
            if word in tokens_vocab:
                weights = torch.FloatTensor(list(map(float, weights)))
                embeddings_matrix[tokens_vocab[word]] = weights

    return embeddings_matrix

def transform_predictions_to_labels(sequence_input_lst, words_mask, idx2label_map, input_type="logit"):
    """
    shape:
        sequence_input_lst: [batch_size, seq_len, num_labels]
        words_mask: [batch_size, seq_len, ]
    """
    words_mask = words_mask.detach().cpu().numpy().tolist()
    if input_type == "logit":
        label_sequence = torch.argmax(torch.nn.functional.softmax(sequence_input_lst, dim=2), dim=2).detach().cpu().numpy().tolist()
    elif input_type == "prob":
        label_sequence = torch.argmax(sequence_input_lst, dim=2).detach().cpu().numpy().tolist()
    elif input_type == "label":
        label_sequence = sequence_input_lst.detach().cpu().numpy().tolist()
    else:
        raise ValueError
    output_label_sequence = []
    for tmp_idx_lst, tmp_label_lst in enumerate(label_sequence):
        tmp_wordpiece_mask = words_mask[tmp_idx_lst]
        tmp_label_seq = []
        for tmp_idx, tmp_label in enumerate(tmp_label_lst):
            if tmp_wordpiece_mask[tmp_idx] != -100:
                tmp_label_seq.append(idx2label_map[tmp_label])
            
        output_label_sequence.append(tmp_label_seq)
        
    return output_label_sequence