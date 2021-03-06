"""

Berger Discrete ZOO
=======================================
(Generating Natural Language Adversarial Examples)
"""

from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.constraints.pre_transformation import (
    InputColumnModification,
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.goal_functions import ZOOUntargetedClassification
from textattack.search_methods import DiscreteZOO
from textattack.shared.attack import Attack
from textattack.transformations import WordSwapEmbeddingStochastic
from textattack.shared.word_embedding import WordEmbedding
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder

from .attack_recipe import AttackRecipe

from textattack.constraints.pre_transformation import RepeatModification, StopwordModification, MaxWordIndexModification, InputColumnModification
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.semantics import WordEmbeddingDistance, BERTScore

STOPWORDS = set([
    'a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against',
    'ain', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although',
    'am', 'among', 'amongst', 'an', 'and', 'another', 'any', 'anyhow', 'anyone',
    'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
    'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below',
    'beside', 'besides', 'between', 'beyond', 'both', 'but', 'by', 'can',
    'cannot', 'could', 'couldn', "couldn't", 'd', 'didn', "didn't", 'doesn',
    "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else',
    'elsewhere', 'empty', 'enough', 'even', 'ever', 'everyone', 'everything',
    'everywhere', 'except', 'first', 'for', 'former', 'formerly', 'from',
    'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
    'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers',
    'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', 'if',
    'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's", 'its', 'itself',
    'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile',
    'mightn', "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must',
    'mustn', "mustn't", 'my', 'myself', 'namely', 'needn', "needn't", 'neither',
    'never', 'nevertheless', 'next', 'no', 'nobody', 'none', 'noone', 'nor',
    'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one',
    'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours',
    'ourselves', 'out', 'over', 'per', 'please', 's', 'same', 'shan', "shan't",
    'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow', 'something',
    'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the',
    'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there',
    'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these',
    'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to',
    'too', 'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon',
    'used', 've', 'was', 'wasn', "wasn't", 'we', 'were', 'weren', "weren't",
    'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter',
    'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether',
    'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose',
    'why', 'with', 'within', 'without', 'won', "won't", 'would', 'wouldn',
    "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
    'your', 'yours', 'yourself', 'yourselves'
])

# Strict Constraints
MAX_LENGTH = 256
COSINE_SIM = 0.9
BERT_SCORE_SIM = 0.9
ALLOW_VERB_NOUN_SWAP = False
TAGGER_TYPE = "flair"

CONSTRAINTS = [
    RepeatModification(),
    StopwordModification(stopwords=STOPWORDS),
    MaxWordIndexModification(max_length=MAX_LENGTH),
    InputColumnModification(["premise", "hypothesis"], {"premise"}),
    WordEmbeddingDistance(min_cos_sim=COSINE_SIM),
    BERTScore(min_bert_score=BERT_SCORE_SIM),
    PartOfSpeech(tagger_type=TAGGER_TYPE,
                 allow_verb_noun_swap=ALLOW_VERB_NOUN_SWAP)
]


class DiscreteZOO2021(AttackRecipe):
  """Berger, N. Ebert, S. Sokolov, A. Riezler, S.
    """

  @staticmethod
  def build(model):

    candidate_number = 10

    attack_embeddings = WordEmbedding.counterfitted_GLOVE_embedding()
    transformation = WordSwapEmbeddingStochastic(
        max_candidates=candidate_number,
        neighborhood_size=1000,
        embedding=attack_embeddings)
    #
    # Don't modify the same word twice or stopwords
    #
    constraints = [RepeatModification(), StopwordModification()]
    #
    # During entailment, we should only edit the hypothesis - keep the premise
    # the same.
    #
    input_column_modification = InputColumnModification(
        ["premise", "hypothesis"], {"premise"})
    constraints.append(input_column_modification)
    #
    # Maximum words perturbed percentage of 20%
    #
    constraints.append(MaxWordsPerturbed(max_percent=0.2))
    #
    # Maximum word embedding euclidean distance of 0.5.
    #
    constraints.append(
        WordEmbeddingDistance(max_mse_dist=100, compare_against_original=False))
    #
    # Goal is untargeted classification
    #
    goal_function = ZOOUntargetedClassification(model)
    #semantic_constraint=UniversalSentenceEncoder(metric="cosine"),
    #interpolation=1.0)

    search_method = DiscreteZOO(candidates=candidate_number,
                                max_changes_per_word=1,
                                max_changes_per_sentence=0,
                                word_embeddings=attack_embeddings)

    return Attack(goal_function, CONSTRAINTS, transformation, search_method)
