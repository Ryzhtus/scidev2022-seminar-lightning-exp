from dataclasses import dataclass
from typing import List
import tqdm


@dataclass
class Word():
    def __init__(self, token, label) -> None:
        self.token = token
        self.label = label

    def __str__(self) -> str:
        return self.token

    def __repr__(self) -> str:
        return f"Word({self.token}, {self.label})"

    def __eq__(self, __o: object) -> bool:
        # compare if two Word() objects are equal. used for feature extraction
        return (self.token == __o.token) & (self.label == __o.label)


@dataclass
class Span():
    def __init__(self) -> None:
        self.span = None
        self.label = None
        self.tokens = []
        self.labels = []

    def __repr__(self) -> str:
        return f"Span({self.span}, {self.label})"

    def __str__(self) -> None:
        return self.span
        
    def add(self, word: Word) -> None:
        if self.span:
            self.span += " "
            self.span += str(word.token)
        else:
            self.span = str(word.token)

        self.tokens.append(word.token)
        self.labels.append(word.label)

        if not self.label:
            if len(word.label) == 1:
                self.label = word.label
            else:
                self.label = word.label[2:]


@dataclass
class Sentence():
    def __init__(self, words: List[Word]) -> None:
        self.words = words
        self.text = ' '.join([word.token for word in self.words])
        self.document_id = None
        self.sentence_id = None
        self.spans = []

        self.__bioes()
        self.__bioes_spans()

    def __len__(self):
        return len(self.words)

    def __repr__(self) -> str:
        return str({"text": self.text,
                    "words": [str(word) for word in self.words],
                    "spans": [str(span) for span in self.spans if span.label != "O"]})

    def __str__(self) -> str:
        return self.text

    def __iter__(self):
        return iter(self.words)

    def __bioes(self) -> None:
        # converts IOB2 into IOBES         
        for idx, word in enumerate(self.words):
            if word.label == "O":
                continue
            elif word.label.split('-')[0] == 'B':
                if idx + 1 != len(self.words) and self.words[idx + 1].label.split('-')[0] == 'I':
                    continue
                else:
                    word.label = word.label.replace('B-', 'S-')
            elif word.label.split('-')[0] == 'I':
                if idx + 1 < len(self.words) and self.words[idx + 1].label.split('-')[0] == 'I':
                    continue
                else:
                    word.label = word.label.replace('I-', 'E-')
            else:
                raise Exception('Invalid IOB format!')

    def __bioes_spans(self) -> None:
        # find IOBES spans in sentence
        for idx in range(len(self.words)):
            word = self.words[idx]

            if word.label == 'O':
                span = Span()
                span.add(word)
                self.__add_span(span)
            elif word.label[0] == "S":
                span = Span()
                span.add(word)
                self.__add_span(span)
            elif word.label[0] == "B":
                span = Span()
                span.add(word)
            elif word.label[0] == "I":
                span.add(word)
            elif word.label[0] == "E":
                span.add(word)
                self.__add_span(span)

    def __bio_spans(self) -> None:
        # find IOB spans in sentence 
        for idx in range(len(self.words) - 1):
            word = self.words[idx]

            if word.label == 'O':
                span = Span()
                span.add(word)
                self.__add_span(span)
            elif word.label[0] == "B":
                span = Span()
                span.add(word)
                if self.words[idx + 1].label == "O":
                    self.__add_span(span)
            elif word.label[0] == "I":
                if self.words[idx + 1].label == "O":
                    span.add(word)
                    self.__add_span(span)
                else:
                    span.add(word)

        if self.words[-1].label == 'O':
            span = Span()
            span.add(self.words[-1])
            self.__add_span(span)
        elif self.words[-1].label[0] == 'B':
            span = Span()
            span.add(self.words[-1])
            self.__add_span(span)
        elif self.words[-1].label[0] == 'I':
            span.add(self.words[-1])
            self.__add_span(span)

    def __add_span(self, span: Span) -> None:
        # add span to the list of sentence's spans
        self.spans.append(span)


@dataclass
class Document():
    def __init__(self, sentences: List[Sentence], document_id: int) -> None:
        self.sentences = sentences
        self.document_id = document_id

        self.document_text = " ".join([sentence.text for sentence in self.sentences])

        for sentence_id, sentence in enumerate(self.sentences):
            sentence.sentence_id = sentence_id
            sentence.document_id = self.document_id

    def __len__(self):
        return len(self.sentences)

    def __iter__(self):
        return iter(self.sentences)