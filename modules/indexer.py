from typing import List
from modules.vocab import Vocab
from modules.document import Document

class Indexer:
    """Индексирует датасет и хранит словари"""
    def __init__(self, lowercase: bool):
        self.token_vocab = Vocab(lowercase=lowercase, paddings=True)

    def index_documents(self, documents: List[Document]):
        """
        Заполняет словари по датасету
        """

        for document in documents:
            for sentence in document:
              self.token_vocab.fill(sentence.words)