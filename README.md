# Google Research Football with Gymnasium and PettingZoo Compatibility

## Features

### Friendly Installation

Update `CMakeLists.txt` and make gfootball easier to install in virtual python environments.

### Compatibility with Gymnasium

- Modify `step` to return `terminated` and `truncated`.
- Modify `reset` to maintain a consistent `np_random` in the environment. Refer to <https://gymnasium.farama.org/api/env/#gymnasium.Env.reset>.

### Compact Representation for Academy Scenarios
See [gfootball/env/\_\_init\_\_.py](./gfootball/__init__.py):

```md
'simplev1': a compact simple representation, adapted from https://github.com/YuriCat/TamakEriFever, which is the implementation of 5th place solution in [gfootball Kaggle Competition](https://www.kaggle.com/c/google-football/discussion/203412).
NOTE: this representation is only designed for cooperative MARL in academy scenarios.
It holds:
    - (x,y) coordinate of current player
    - (x,y) direction of current player
    - (is_sprinting, is_dribbling) agent status
    - (Δx,Δy) relative coordinates of other left team players, size (n1-1) * 2 
    - (Δx,Δy) relative coordinates of right team players, size n2 * 2
    - (Δx,Δy) relative coordinate of current player to the ball
    - (x,y) coordinates of other left team players, size (n1-1) * 2 
    - (x,y) direction of other left team players, size (n1-1) * 2
    - (x,y) coordinates of right team players, size n2 * 2
    - (x,y) direction of right team players, size n2 * 2 
    - (x,y,z) ball position
    - (Δx,Δy,Δz) ball direction
    - one hot encoding of ball ownership (noone, left, right)
    - one hot encoding of `game_mode`
    - one hot encoding of which player is active (agent id), size n1
Total dim:
    4 * 2 + (n1-1) * 2 * 3 + n2 * 2 * 3 + 3 + 3 + 3 + n1 + 7
    = 7 * n1 + 6 * n2 + 18
```

## Installation

```shell
# system dependence
sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev \
libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
libdirectfb-dev libst-dev mesa-utils xvfb x11vnc
# pip dependence
pip install -U pip setuptools psutil wheel
conda install anaconda::py-boost -y
# install gfootball
pip install git+https://github.com/xihuai18/gfootball-gymnasium.git
```
Other installation problems may be found in the original [README](https://github.com/google-research/football).

`libffi.so.7` may not be included in your environment variables, adding it before running gfootball.

```shell
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7 
```

## Test

```shell
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7 
python -c "import gymnasium as gym; import gfootball; env = gym.make('GFootball/academy_3_vs_1_with_keeper-simplev1-v0'); print(env.reset()); print(env.step([0]))"
```