# Single agent PPO

## Environment

The reacher environment (continuous 33 dimensional state space and continuous 4 dimensional action space). The environment is considered solved when the agent get an average score of 30 over 100 consecutive episodes.

## Installing the environment

* Download the environment:
  * Linux: **[download](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)**
  * Mac OSX: **[download](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)**
  * Windows (32-bit): **[download](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)**
  * Windows (64-bit): **[download](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)**
* Unzip in folder p2_continuous-control.
* In the code, import the UnityEnvironment as follow (the file_name should target the reader's own *Reacher.exe*):

    ```python
    from unityagents import UnityEnvironment
    env = UnityEnvironment(file_name="C:\\Users\AL\Documents\GitHub\deep-reinforcement-learning\p2_continuous-control\Reacher_Windows_x86_64\Reacher.exe", no_graphics=True)

    ```

## Instructions

Run the PPO_single_agent.py to train the agent. After being trained over 2000 episodes or if the environment is solved, the code will plot the scores and the average score over the last 100 episodes. It will save the neural network weights in *network.pth*. The scores will be saved in *scores.txt*. The code writes the current episode, the average score over the last 100 episodes, the maximum score and the current standard deviation. The agent should be able to solve the environment in approximatively 900 episodes.
