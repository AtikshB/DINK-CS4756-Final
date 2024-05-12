# DINK: Differently Initialized Q-Networks
## (CS4756 Robot Learning Final Project)

A project that analyzes whether different dataset initialization affects the final performance of a Q-Network.

## Atari Grand Challenge Dataset
Download the SpaceInvaders zip file containing screens and trajectories, and place folder at top level of the repository:
https://github.com/yobibyte/atarigrandchallenge

## Installation
Create a new conda or python virtual environment in 3.10.12 and run below commands:
~~~
pip install torch==2.2.0
pip install gym[atari,accept-rom-license]==0.21.0
pip install numpy
pip install matplotlib
pip install tqdm
pip install opencv-python
~~~
