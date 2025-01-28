# Tutorial: Integrating Custom Models in ACEGEN

The tricky thing with reinforcement learning is there are often many hyperparameters and configurations that can be tuned. Wandb can be conveniently used with hydra (used to parse the yaml files in acegen) out of the box to conduct hyperparameter optimization. This consists of two steps,
1. Defining a sweep (what hyperparameters to test)
2. Starting agents (a controller that tries different configurations)

For more details and further options not mentioned here, see the wandb tutorial [here](https://docs.wandb.ai/guides/sweeps). 

## Hyperparameter optimization of the PPO algorithm

### Defining the sweep

First we need to define a YAML file that tells wandb what script we are running, how to search hyperparameters, and what hyperparameters to search.
Here is a simple example called `ppo_sweep.yaml`,
```YAML
program: ppo.py 
method: random 
parameters:
  num_envs:
    values: [32, 64, 128, 256]
  critic_coef:
    values: [0.1, 0.25, 0.5]
  entropy_coef:
    values: [0.01, 0.05]
  ppo_epochs:
    values: [1, 2, 3]

command:
  - ${env}
  - python
  - ${program}
  - ${args_no_hyphens} # Necessary for use with hydra as used in ACEGEN
```

Now we have our sweep configuration, we start our sweep in the CLI with the following command,

```
wandb sweep ppo_sweep.yaml 
```

This will return a `sweep_id` which is needed to run the agents in the next step.

### Run agents
Now we have a sweep setup, we can start agents that sample a set of hyperparameters from our sweep setup and run the program with them.

(Quite literally by looping over the command specified in the sweep, for example, `python ppo.py num_envs=32 critic_coef=0.25 entropy_coef=0.05 ppo_epochs=3`)

To start an agent, simply run the following command,

```
wandb agent --count <number_of_runs> sweep_id
```

This command can be run several times to create as many agents as you wish (for example, on different GPUs), enabling parallel agents testing different hyperparameters.

## Things to note

- Wandb provides an awesome GUI online to measure and track the best configurations according to some metrics (have you logged them in the script??)
- Any hyperparameters not specified in the sweep configuration will be taken from the default configuration path.
  - This can be changed by adding `--config-name=<new_default.yaml>` to the command section of the sweep inbetween `${program}` and `${args_no_hyphens}`.
  - Or by changing the default path specified at the top of main in the respective script.
- Something may not be interpreted correctly, specifically `null` may be interpreted as a string. So it's recommended to use `False` or `0`.
