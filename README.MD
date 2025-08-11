# Cryo-RL: automating prostate cancer cryoablation planning with reinforcement learning

An open source repository for implementing Reinforcement learning (RL) to learn needle placement planning for prostate cryoablation

# Environment 

Create an environment with the following dependancies using the following command:

```
conda env create -f environment.yml
```

# Experiments 
To run the sparse reward experiment use the following script:

```
python Sparse_exp_P1.py
```

To run the dense reward experiment use the following script:

```
python Dense_exp_P1.py
```
# Evaluation 

To create the grid visualisatio for any experiment, run the following code 

```
python eval.py
```


