python3 ./analysis/success_over_budget.py \
    "none_constraints_success_over_budget_mr.pdf" \
    "None Constraints, MR" \
    DiscreteZOO ./zoo_grid_search_results/bert-mr-test/zoountargeted/17_seed41.csv \
    Greedy ./results/bert-mr-test/word-swap-embedding/none/zoountargeted/greedy.csv \
    Beam=4 ./results/bert-mr-test/word-swap-embedding/none/zoountargeted/beam4.csv \
    GreedyWIR_Delete ./results/bert-mr-test/word-swap-embedding/none/zoountargeted/greedyWIR_delete.csv \
    GreedyWIR_Unk ./results/bert-mr-test/word-swap-embedding/none/zoountargeted/greedyWIR_unk.csv \
    GreedyWIR_Random ./results/bert-mr-test/word-swap-embedding/none/zoountargeted/greedyWIR_random.csv \
    Genetic ./results/bert-mr-test/word-swap-embedding/none/zoountargeted/genetic.csv 

python3 ./analysis/success_over_budget.py \
    "none_constraints_success_over_budget_snli.pdf" \
    "None Constraints, SNLI" \
    DiscreteZOO ./zoo_grid_search_results/bert-snli-test/zoountargeted/17_41.csv \
    Greedy ./results/bert-snli-test/word-swap-embedding/none/zoountargeted/greedy.csv \
    Beam=4 ./results/bert-snli-test/word-swap-embedding/none/zoountargeted/beam4.csv \
    GreedyWIR_Delete ./results/bert-snli-test/word-swap-embedding/none/zoountargeted/greedyWIR_delete.csv \
    GreedyWIR_Unk ./results/bert-snli-test/word-swap-embedding/none/zoountargeted/greedyWIR_unk.csv \
    GreedyWIR_Random ./results/bert-snli-test/word-swap-embedding/none/zoountargeted/greedyWIR_random.csv \
    Genetic ./results/bert-snli-test/word-swap-embedding/none/zoountargeted/genetic.csv 