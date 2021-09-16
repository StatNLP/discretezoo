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


def Attack(model):
  goal_function = textattack.goal_functions.ZOOUntargetedClassification(model)
  search_method = textattack.search_methods.RandomWordSwapWIR(
      wir_method="random", improvement_only=True, loop_until_success=True, budget=30)
  transformation = TRANSFORMATION
  constraints = CONSTRAINTS
  #Remove repeat modification from constraints
  for index, constraint in enumerate(constraints):
      if isinstance(constraint, textattack.constraints.pre_transformation.RepeatModification):
          del constraints[index]
  return textattack.shared.Attack(goal_function, constraints, transformation,
                                  search_method)
