# Deep Reinforcement Learning - Continuous Control Project

Implementation of continuous action-space [Proximal Policy Optimization (PPO)](https://openai.com/blog/openai-baselines-ppo/) agent for "Continuous Control" project in Udacity's [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

By Sebastian Castro, 2020

---

## Project Introduction

This project uses the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment from Unity ML-Agents.

This environment consists of 20 identical simulated robot arms which must place their end effector inside spheres that move around them. The spheres, which are normally blue, are colored green when the arms are positioned inside them. The arms have two joints with 2 degrees of freedom each, which can be actuated with torques.

![Environment animation](media/openai_reacher.gif)

The specifics of the environment are:

* **State:** `33` variables corresponding to position, rotation, velocity, and angular velocities of the arm.
* **Actions:** A vector with `4` elements, with each element corresponding to joint torques that can have any continuous value between `-1.0` and `1.0`.
* **Reward:** The agent receives `+0.1` reward each time step that the arm's end effector is inside the target goal location defined by the sphere around it.

As per the project specification, an agent is considered to have "solved" the problem if the average reward over all the agents exceeds `30` by the end of an episode.

To see more details about the PPO agent implementation, and training results, refer to the [Report](Report.md) included in this repository.

---

## Getting Started

To get started with this project, first you should perform the setup steps in the [Udacity Deep Reinforcement Learning Nanodegree Program GitHub repository](https://github.com/udacity/deep-reinforcement-learning). Namely, you should

1. Install [Conda](https://docs.conda.io/en/latest/) and create a Python 3.6 virtual environment
2. Install [OpenAI Gym](https://github.com/openai/gym)
3. Clone the [Udacity repo]((https://github.com/udacity/deep-reinforcement-learning)) and install the Python requirements included
4. Download the Reacher Unity files appropriate for your operating system and architecture ([Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip), [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip), [Win32](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip), [Win64](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip))

Once you have performed this setup, you should be ready to run the [`reacher_ppo.ipynb`](reacher_ppo.ipynb) Jupyter Notebook in this repo. This notebook contains all the steps needed to define and train a DQN Agent to solve this environment.