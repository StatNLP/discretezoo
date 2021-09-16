"""
scikit-learn Model Wrapper
--------------------------
"""

import pandas as pd

from .model_wrapper import ModelWrapper


class SklearnModelWrapper(ModelWrapper):
  """Loads a scikit-learn model and tokenizer (tokenizer implements
    `transform` and model implements `predict_proba`).

    May need to be extended and modified for different types of
    tokenizers.
    """

  def __init__(self, model, tokenizer):
    self.model = model
    self.tokenizer = tokenizer
    self.call_count = 0

  def reset_call_count(self):
    self.call_count = 0

  @property
  def get_call_count(self):
    return self.call_count

  def __call__(self, text_input_list):
    self.call_count += len(text_input_list)
    encoded_text_matrix = self.tokenizer.transform(text_input_list).toarray()
    tokenized_text_df = pd.DataFrame(encoded_text_matrix,
                                     columns=self.tokenizer.get_feature_names())
    return self.model.predict_proba(tokenized_text_df)

  def get_grad(self, text_input):
    raise NotImplementedError()
