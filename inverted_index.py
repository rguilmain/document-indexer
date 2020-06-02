import math
import os
import pickle
from collections import Counter, defaultdict


class Index:
    """
    An inverted index is a map from terms to collections of (document, term count) pairs. E.g.:
    index = {
        'quick': [('doc1', 2), ('doc3', 1)],
        'brown': [('doc2', 3)],
        'fox': [('doc1', 1), , ('doc2', 1), ('doc3', 1)]
    }
    This allows for quick scoring and retrieval of documents given a query.
    Term counts are also commonly referred to as 'term frequencies'.
    """

    def __init__(self):
        self.index = defaultdict(list)
        self.doc_lengths = {}

    @property
    def num_docs(self) -> int:
        return len(self.doc_lengths)

    @property
    def avg_doc_len(self) -> float:
        return sum(self.doc_lengths.values()) / self.num_docs

    def index_document(self, filepath: str) -> None:
        doc_name, _ = os.path.splitext(os.path.basename(filepath))
        print(f'indexing {doc_name}...')
        term_counts = Counter()
        with open(filepath, 'r', encoding='utf8') as f_in:
            for line in f_in:
                for term in line.strip().split():
                    term_counts[term] += 1
        for term, count in term_counts.items():
            self.index[term].append((doc_name, count))
        self.doc_lengths[doc_name] = sum(term_counts.values())

    def index_directory(self, directory: str) -> None:
        num_indexed = 0
        for filename in os.listdir(directory):
            if not filename.endswith('.txt'):
                print(f'skipping {filename}...')
                continue
            filepath = os.path.join(directory, filename)
            self.index_document(filepath)
            num_indexed += 1
        print(f'indexed {num_indexed} documents')

    def save_to_file(self, filepath: str) -> None:
        print(f'writing index to {filepath}...')
        pickle.dump((self.index, self.doc_lengths), open(filepath, 'wb'))

    def load_from_file(self, filepath: str) -> None:
        print(f'loading index from {filepath}...')
        self.index, self.doc_lengths = pickle.load(open(filepath, 'rb'))

    def display(self) -> None:
        print('index contents:')
        left_pad = max(len(term) for term in self.index) + 2
        for term, doc_counts in sorted(self.index.items()):
            print(f'{term:>{left_pad}}: {doc_counts}')

    def idf(self, term: str) -> float:
        # For x from 0 to 1, -log(x):
        #   1) goes to positive infinity as x goes to 0, and
        #   2) is 0 when x is 1.
        return -math.log(len(self.index[term]) / self.num_docs)

    def query_tfidf(self, query: str) -> dict:
        doc_scores = defaultdict(float)
        for term in query.strip().split():
            for doc, tf in self.index[term]:
                doc_scores[doc] += tf * self.idf(term)
        return self._rank(doc_scores)

    def query_bm25(self, query: str, *, k1: float=2.0, b: float=0.75) -> dict:
        doc_scores = defaultdict(float)
        for term in query.strip().split():
            for doc, tf in self.index[term]:
                doc_len_norm = (k1 + 1.0) / (tf + k1 * (1.0 - b + b * (self.doc_lengths[doc] / self.avg_doc_len)))
                doc_scores[doc] += tf * self.idf(term) * doc_len_norm
        return self._rank(doc_scores)

    def _rank(self, doc_scores: dict) -> dict:
        return {doc: score for doc, score
                in sorted(doc_scores.items(), key=lambda item: -item[1])}
