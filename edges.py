import numpy as np
from abc import ABC, abstractmethod
from gensim.models import KeyedVectors
from itertools import combinations_with_replacement
from functools import reduce


class EdgeEmbedder(ABC):

    def __init__(self, keyed_vectord: KeyedVectors, quiet: bool = False):
        """

        :param keyed_vectord:
        :param quiet:
        """

        self.kv = keyed_vectord
        self.quiet = quiet

    @abstractmethod
    def _embed(selfself, edge: tuple) -> np.ndarray:
        """

        :param edge: tuple of two nodes
        :return: Edge embedding
        """
        pass

    def __getitem__(self, edge) -> np.ndarry:
        if not isinstance(edge, tuple) or not len(edge) == 2:
            raise ValueError('edge must be a tuple of two nodes')

        if edge[0] not in self.kv.index2word:
            raise KeyError('node {} does not exist in given KeyedVectors'.format(edge[0]))

        if edge[1] not in self.kv.index2word:
            raise KeyError('node {} does not exist in given KeyedVectors'.format(edge[1]))

        return self._embed(edge)

    def as_keyed_vectors(self) -> KeyedVectors:
        """
        Generated a KeyedVectors instance with all the possible edge embeddings
        :return: Edge embeddings
        """

        edge_generator = combinations_with_replacement(self.kv.index2word, r=2)

        if not self.quiet:
            vocab_size = len(self.kv.vocab)
            total_size = reduce(lambda x, y: x * y, range(1, vocab_size +2)) / \
                         (2 * reduce(lambda x, y: x * y, range(1, vocab_size)))
