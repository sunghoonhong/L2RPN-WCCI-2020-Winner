# Learning To Run a Power Network (L2RPN) WCCI 2020 Competition Winner Code
This is a repository for L2RPN WCCI 2020 competition during June 2020 - July 2020.
The RL agent based on this repository won this competition.


[Competition Homepage](https://l2rpn.chalearn.org)


[Competition Result](https://competitions.codalab.org/competitions/24902#results) (Click 'Test phase' button)


# Summary
This competition aims to train an agent that is able to run a power network as long as possible and minimize power loss at the same time.
We utilize bus switching action so that the agent configures topology of the power network. This repository presents an agent based on Soft Actor-Critic and Graph Neural Networks. Furthermore, we introduce a hierarchical framework for temporal abstraction due to the environmental constraints.

# Environment
```
Python==3.7.7  
PyTorch==1.5.0  
Grid2Op==0.9.4  (IMPORTANT)
```
```
pip install -r requirements.txt
```

# Usage
## Make submission for the competition
```
python check_your_submission.py
```


The HTML file 'results/results.html' illustrates thee result for local dataset.
You can change hyperparameters in ```params.json```

## NOTE
This code is available on ```l2rpn_wcci_2020``` only in ```Grid2Op==0.9.4``` version, which is used in the competition.
It requires ```mean.pt``` and ```std.pt```, which is statistics of randomly collected observations. (```data/mean.pt```, ```data/std.pt```)
If you have those files, it is possible to deploy on other grids. (```evaluate``` only)  
If you need a training method, you may refer to [here](https://github.com/sunghoonhong/SMAAC).

# License Information
Copyright (c) 2020 KAIST-AILab

This source code is subject to the terms of the Mozilla Public License (MPL) v2 also available [here](https://www.mozilla.org/en-US/MPL/2.0/)


**We do NOT allow commercial use of this code.**
