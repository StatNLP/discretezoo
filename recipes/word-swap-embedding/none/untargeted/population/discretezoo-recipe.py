"""

Berger Discrete ZOO
=======================================
(Generating Natural Language Adversarial Examples)
"""

import textattack
import os, sys
current_dir = os.path.dirname(os.path.realpath(__file__))
constraint_dir = os.path.normpath(
    os.path.join(current_dir, os.pardir, os.pardir))
transformation_dir = os.path.normpath(
    os.path.join(current_dir, os.pardir, os.pardir, os.pardir))
sys.path.append(constraint_dir)
sys.path.append(transformation_dir)
from transformation import TRANSFORMATION
from constraint import CONSTRAINTS

from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.constraints.pre_transformation import (
    InputColumnModification,
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import DiscreteZOO
from textattack.transformations import WordSwapEmbeddingDisplacement
from textattack.shared.word_embedding import WordEmbedding


def Attack(model):
    """Berger, N. Ebert, S. Sokolov, A. Riezler, S.
  """
    candidate_number = 10

    attack_embeddings = WordEmbedding.counterfitted_GLOVE_embedding()

    transformation = WordSwapEmbeddingDisplacement(
        max_candidates=10, embedding=attack_embeddings, learning_rate=0.1, discretize_by_cosine=True)

    constraints = CONSTRAINTS

    goal_function = UntargetedClassification(model, use_cache=True)

    search_method = DiscreteZOO(candidates=candidate_number,
                                neighborhood_multiplier=1,
                                max_changes_per_word=10,
                                max_changes_per_sentence=0,
                                word_embeddings=attack_embeddings,
                                average_displacements=True,
                                sample_cos_nn=True)

    return textattack.shared.Attack(goal_function, constraints, transformation,
                                    search_method)
