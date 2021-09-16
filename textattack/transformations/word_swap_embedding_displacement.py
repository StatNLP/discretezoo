"""
Word Swap by Embedding
============================================

Based on paper: `<arxiv.org/abs/1603.00892>`_

Paper title: Counter-fitting Word Vectors to Linguistic Constraints

"""
import random

from textattack.shared import AbstractWordEmbedding, WordEmbedding
from textattack.transformations.word_swap import WordSwap


class WordSwapEmbeddingDisplacement(WordSwap):
  """Transforms an input by replacing its words with synonyms in the word
    embedding space.

    Args:
        max_candidates (int): maximum number of synonyms to pick
        embedding (textattack.shared.AbstractWordEmbedding): Wrapper for word embedding
    """

  def __init__(self,
               max_candidates=50,
               embedding=WordEmbedding.counterfitted_GLOVE_embedding(),
               learning_rate=1.0,
               discretize_by_cosine=False,
               **kwargs):
    super().__init__(**kwargs)
    self.max_candidates = max_candidates
    if not isinstance(embedding, AbstractWordEmbedding):
      raise ValueError(
          "`embedding` object must be of type `textattack.shared.AbstractWordEmbedding`."
      )
    self.embedding = embedding
    self.learning_rate = learning_rate
    self.discretize_by_cosine = discretize_by_cosine

  def _get_replacement_words(self, word, displacement=None, **kwargs):
    """Returns a list of possible 'candidate words' to replace a word in a
        sentence or phrase.

        Based on nearest neighbors selected word embeddings.
        """
    if displacement is None:
      raise ValueError("Displacement can not be None for this transformation.")

    try:
      word_id = self.embedding.word2index(word.lower())
      current_embedding = self.embedding[word_id]
      updated_position = current_embedding + self.learning_rate * displacement
      if self.discretize_by_cosine:
        return self.embedding.get_cos_nn(updated_position, self.max_candidates)[0]
      else:
        return self.embedding.get_euc_nn(updated_position, self.max_candidates)[0]
    except KeyError:
      # This word is not in our word embedding database, so return an empty list.
      return []

  def extra_repr_keys(self):
    return ["max_candidates", "embedding", "learning_rate", "discretize_by_cosine"]

  @property
  def deterministic(self):
    return False

def recover_word_case(word, reference_word):
  """Makes the case of `word` like the case of `reference_word`.

    Supports lowercase, UPPERCASE, and Capitalized.
    """
  if reference_word.islower():
    return word.lower()
  elif reference_word.isupper() and len(reference_word) > 1:
    return word.upper()
  elif reference_word[0].isupper() and reference_word[1:].islower():
    return word.capitalize()
  else:
    # if other, just do not alter the word's case
    return word

