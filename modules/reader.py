from typing import List
from modules.document import Word, Sentence, Document
from tqdm import tqdm

class ReaderCoNLL():
    def __parse(self, filename: str):
        rows = open(filename, 'r').read().strip().split("\n\n")
        sentences = []
        documents = []
        document_sentence = []

        document_id = -1

        for sentence in tqdm(rows):       
            for line in sentence.splitlines():
                token = line.split()[0]
                label = line.split()[-1]       
                word = Word(token, label)

                if token == "-DOCSTART-":
                    if document_id == -1:
                        document_id += 1
                    else:
                        documents.append(Document(sentences, document_id))
                        document_id += 1
                        sentences = []
                    
                document_sentence.append(word)
            
            sentences.append(Sentence(document_sentence))
            document_sentence = []

        # don't forget to add the last document (because there is no -DOCSTART- tag in the ned of a file)
        documents.append(Document(sentences, document_id))

        return documents

    def parse(self, filename: str) -> List[Document]:
        return self.__parse(filename)