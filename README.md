# MEow Replication

## Contributions

In this project, we replicated "Maximum Entropy Reinforcement Learning via
Energy-Based Normalizing Flow", reimplementing the flow policy and training process ourselves.

Our main code resides in [meow_ours.py](meow_ours.py), [meow_ours_robust.py](meow_ours_robust.py), [flow_policy.py](ebflows/flow_policy.py), and [td3_ours_robust.py](td3_ours_robust.py). However, we also improved the documentation of code within [ebflows](ebflows), and wrote scripts in [figure_creation](figure_creation) that may be helpful for generating future charts or recreating our figures.

# Instructions

### MuJoCo Environments (Ant-v4, Humanoid-v4, HumanoidStandup-v4)

**We wrote this section of the code for Python 3.9, to match the MEow paper.**

To start, create a new conda environment (called `meow`) for Python 3.9 and activate it: `conda activate meow`

Then, run `bash setup.bash` within the conda environment.

MEow seeds (and other hyperparameters) should be changed within the corresponding configuration files.

To run Ant-v4 with MEow:
```
python meow_ours.py --config "config_meow_antv4.yaml"
```

To run Ant-v4 with SAC:
```
python sac_continuous_action.py --seed <SEED> --env-id Ant-v4 --total-timesteps 4000000 --tau 0.0001 --alpha 0.05 --learning_starts 5000
```

To run Humanoid-v4 with MEow:
```
python meow_ours.py --config "config_meow_humanoidv4.yaml"
```

To run Humanoid-v4 with SAC:
```
python sac_continuous_action.py --seed <SEED> --env-id Humanoid-v4 --total-timesteps 5000000 --tau 0.0005 --alpha 0.125 --learning_starts 5000
```

To run HumanoidStandup-v4 with MEow:
```
python meow_ours.py --config "config_meow_humanoid_standupv4.yaml"
```

To run HumanoidStandup-v4 with SAC:
```
python sac_continuous_action.py --seed <SEED> --env-id HumanoidStandup-v4 --total-timesteps 2500000 --tau 0.0005 --alpha 0.125 --learning_starts 5000
```

### MuJoCo Environments (AntRandom-v5)

**We wrote this section of the code for Python 3.12, for compatibility with Adroit.**

On Adroit, run `bash setup.bash`.

MEow and TD3 seeds (and other hyperparameters) should be changed within the corresponding configuration files.

To run AntRandom-v5 with MEow:
```
python meow_ours_robust.py --config "config_meow_ant_randomv5.yaml"
```

To run AntRandom-v5 with TD3:
```
python td3_ours_robust.py --config "config_td3_ant_randomv5.yaml"
```