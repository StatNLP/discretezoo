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
from textattack.goal_functions import ZOOUntargetedClassification
from textattack.search_methods import DiscreteZOO
from textattack.transformations import WordSwapEmbeddingDisplacement
from textattack.shared.word_embedding import WordEmbedding


def Attack(model):
    """Berger, N. Ebert, S. Sokolov, A. Riezler, S.
  """
    candidate_number = 25

    attack_embeddings = WordEmbedding.counterfitted_GLOVE_embedding()

    transformation = WordSwapEmbeddingDisplacement(
        max_candidates=25, embedding=attack_embeddings, learning_rate=10.0, discretize_by_cosine=True)

    constraints = CONSTRAINTS

    goal_function = ZOOUntargetedClassification(model, use_cache=True)

    search_method = DiscreteZOO(candidates=candidate_number,
                                neighborhood_multiplier=2,
                                max_changes_per_word=1,
                                max_changes_per_sentence=0,
                                word_embeddings=attack_embeddings,
                                average_displacements=False,
                                sample_cos_nn=True,
                                normalize_differences=True,
                                normalize_displacements=True)

    return textattack.shared.Attack(goal_function, constraints, transformation,
                                    search_method)
