"""
TensorFlow Model Wrapper
--------------------------
"""

import numpy as np

from .model_wrapper import ModelWrapper


class TensorFlowModelWrapper(ModelWrapper):
  """Loads a TensorFlow model and tokenizer.

    TensorFlow models can use many different architectures and
    tokenization strategies. This assumes that the model takes an
    np.array of strings as input and returns a tf.Tensor of outputs, as
    is typical with Keras modules. You may need to subclass this for
    models that have dedicated tokenizers or otherwise take input
    differently.
    """

  def __init__(self, model):
    self.model = model
    self.call_count = 0

  def reset_call_count(self):
    self.call_count = 0

  @property
  def get_call_count(self):
    return self.call_count

  def __call__(self, text_input_list):
    self.call_count += len(text_input_list)
    text_array = np.array(text_input_list)
    preds = self.model(text_array)
    return preds.numpy()

  def get_grad(self, text_input):
    raise NotImplementedError()
