# Reinforcement learning for Brain Maturation Modeling
This repo aims to establish a reinforcement learning (RL) framework to model brain maturation process. 

Relevant projects can be found in the following links: <a href="https://github.com/xinzhoucs/RNN_BrainMaturation">RNN_BrainMaturation</a>, <a href="https://github.com/gyyang/multitask">Multitask</a> and <a href="https://github.com/nmasse/Short-term-plasticity-RNN">Short-term-plasticity-RNN</a>.

More details and updates are coming -- Jiajie Xiao.  


## Installation

1. Clone RL_brainMaturation repository
```
git clone https://github.com/jiajiexiao/RL_BrainMaturation.git
cd RL_BrainMaturation/
```

2. Install in development mode
### Install in “develop” or “editable” mode:
```
python setup.py develop
```
or
```
pip install -e ./
```

3. Usage
```
import rlbrainmaturation
from rlbrainmaturation.tasks.task import Task, Instruction
from rlbrainmaturation.tasks.odr import ODR
from rlbrainmaturation.envs.environment import Environment
...
```