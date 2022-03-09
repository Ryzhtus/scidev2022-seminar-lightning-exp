from typing import List
from modules.document import Document
from modules.indexer import Indexer

import torch 
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
 
class CoNLLDataset(Dataset):
    def __init__(self, documents: List[Document], indexer: Indexer):
        self.documents = documents
        self.sentences = [sentence for document in self.documents for sentence in document]
        self.indexer = indexer
        
        self.entity_tags = sorted(list(set(word.label for sentence in self.sentences for word in sentence)))
        self.tag2idx = {tag: idx for idx, tag in enumerate(self.entity_tags)}
        self.idx2tag = {idx: tag for idx, tag in enumerate(self.entity_tags)}

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        sentence = self.sentences[item]

        tokenized_words = []
        tokenized_labels = []

        for word in sentence:
            tokenized_words.append(self.indexer.token_vocab[word.token])
            tokenized_labels.append(self.tag2idx[word.label])

        words_mask = [1 for _ in range(len(tokenized_words))]        

        return torch.LongTensor(tokenized_words), torch.LongTensor(tokenized_labels), torch.LongTensor(words_mask)
        

    def paddings(self, batch):
        input_ids, labels, words_mask = list(zip(*batch))

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        words_mask = pad_sequence(words_mask, batch_first=True, padding_value=-100)

        return {"input_ids": input_ids, 
                "labels": labels,
                "words_mask": words_mask
                }