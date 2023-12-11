# Promoting Counterfactual Robustness through Diversity


This repository contains the code used for the AAAI24 paper:

```
Promoting Counterfactual Robustness through Diversity
```

All the experiments can be replicated using the <code>./scripts/run_all_set.sh</code> (requires Python 3).


The full list of parameters can be obtained by running:

```
python main.py -h
```

For example, the following can be used to generate counterfactual explanations:

```
python main.py german ../datasets/german/ ../models/ ../results/example.txt ours --norm 1 --alpha 1000 --beta 0.1 --gamma 0.1 --opt
```

