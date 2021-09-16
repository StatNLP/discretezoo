import os
import run_experiment

MODELS = ["bert-base-uncased-mr", "bert-base-uncased-snli"]
MODEL_RESULT = {"bert-base-uncased-mr": "bert-mr-test",
                "bert-base-uncased-snli": "bert-snli-test"}
TRANSFORMATIONS = ["word-swap-embedding"]
CONSTRAINT_LEVEL = ["strict", "lax", "none"]
SEARCH_METHODS = {
    "beam-search": ["greedy", "beam4", "beam8"],
    "greedy-word-wir": ["delete", "unk", "pwws", "random", "closest-improvement-only", "furthest-improvement-only", "random-choice-improvement-only-continued-sampling", "random-choice"],
    "population": ["discretezoo", "genetic", "pso"] 
}

GOAL_FUNCTIONS = ["zoountargeted", "untargeted"]

SEED_BANK = [41, 42, 43, 44, 45, 46, 47]

print(f"Running experiment for model \"{MODELS}\"")

for model in MODELS:
  for transformation in TRANSFORMATIONS:
    for constraint in CONSTRAINT_LEVEL:
      for family in SEARCH_METHODS:
        for search in SEARCH_METHODS[family]:
          for seed in SEED_BANK:
            for goal_function in GOAL_FUNCTIONS:
              recipe_path = f"recipes/{transformation}/{constraint}/{goal_function}/{family}/{search}-recipe.py"
              result_file_name = f"greedyWIR_{search}" if family == "greedy-word-wir" else search
              exp_base_name = f"{MODEL_RESULT[model]}/{transformation}/{constraint}/{goal_function}"
              result_dir = f"grid_results/{exp_base_name}"
              chkpt_dir = f"grid-end-checkpoints/{exp_base_name}"
              if not os.path.exists(result_dir):
                os.makedirs(result_dir)
              if not os.path.exists(chkpt_dir):
                os.makedirs(chkpt_dir)

              result_file_name = result_file_name + f"seed{seed}"

              log_txt_path = f"{result_dir}/{result_file_name}.txt"
              log_csv_path = f"{result_dir}/{result_file_name}.csv"
              chkpt_path = f"{exp_base_name}/{result_file_name}"

              run_experiment.run(model,
                                recipe_path,
                                log_txt_path,
                                log_csv_path,
                                chkpt_path=chkpt_path,
                                random_seed=seed)
