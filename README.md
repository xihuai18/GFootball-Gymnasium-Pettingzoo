# Google Research Football with Gymnasium and PettingZoo Compatibility

Update `CMakeLists.txt` and make gfootball easier to install in virtual python environments.

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
python -c "import gymnasium as gym; import gfootball; env = gym.make('GFootball/academy_3_vs_1_with_keeper-simple115v2-v0'); print(env.reset()); print(env.step([0]))"
```