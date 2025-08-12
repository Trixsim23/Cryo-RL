# Cryo-RL: automating prostate cancer cryoablation planning with reinforcement learning

An open source repository for implementing Reinforcement learning (RL) to learn cryoprobe placement planning for prostate cryoablation


## Abstract 

Cryoablation is a minimally invasive localised treatment for prostate cancer that destroys malignant tissue by freezing, while sparing surrounding healthy structures. Its success depends on accurate preoperative planning of cryoprobe placements to fully cover the tumour and avoid critical anatomy. This planning is currently manual, expertise-driven, and time-consuming, leading to variability in treatment quality and limited scalability. 
In this work, we introduce Cryo-RL, a reinforcement learning framework that models cryoablation planning as a Markov decision process and learns an optimal policy for cryoprobe placement. Within a simulated environment that models clinical constraints and stochastic intraoperative variability, an agent sequentially selects cryoprobe positions and ice sphere diameters. Guided by a reward function, based on tumour coverage, this agent learns a cryoablation strategy that leads to optimal cryoprobe placements without the need for any manually-designed plans.
Evaluated on 583 retrospective prostate cancer cases, Cryo-RL achieved over 8 percentage-point Dice improvements compared with the best automated baselines and matched human expert performance while requiring substantially less planning time. These results highlight the potential of reinforcement learning to deliver clinically viable, reproducible, and efficient cryoablation plans.

## Environment 

Create an environment with the following dependancies using the following command:

```
conda env create -f environment.yml
```

## Experiments 
To run the non-shaped reward experiment use the following script:

```
python shaped_exp_P1.py
```

To run the shaped reward experiment use the following script:

```
python shaped_exp_P1.py
```

## Human Annotation Tool 
 
 A collection tool which displays MRI slices from a dataset which has contains .NIfTi files including a 3D MRI volume and labelled prostate and lesion masks. The code allows user to collect placement data by interacting with loaded patients. Other features include exporting a summary of overall results after a given run and .csv export of created placements to use for comparison with other RL made placements. This tool can be found in the collection tool and can be run using the following script:

 ```
 python ./collection_tool/collection_tool.py 
 ```


