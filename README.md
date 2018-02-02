# simple_ppo

Implementation of PPO in `pytorch`.

(Mostly) from scratch, but inspired by:
 
- [pytorch-ppo](https://github.com/tpbarron/pytorch-ppo)
- [openai/baselines](https://github.com/openai/baselines)
- [pytorch-a2c-ppo-acktr](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr)

### Requirements

Requires `pytorch` + `openai/gym` (w/ `mujoco`, for continuous control experiments).

### Installation

```
conda create --yes -n simple_ppo_env python=3.6 anaconda
source activate simple_ppo_env

# Install pytorch
conda install -y pytorch torchvision -c pytorch

# Install openai gym
pip install gym
conda install -y -c conda-forge opencv 

```

### Usage

See `run.sh` for usage.