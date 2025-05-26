# multiprocessing_PPO
In this repository, I a Proximal Policy Optimization algorithm is presented. The environment is in the file kondili.py and needs to be set as a gym environment. The training is configured to distribute the collection of data with a multiprocess method. The network uses an attention module to gain insight from the environment. The agent is a hybrid agent that outputs multiple actions at every time-interval in the horizon. The environment is a multi-product batch chemical process.

To run this code, open a terminal (can be at visual studio code), locate the PATH, open it and write "python main_daniel.py" in the terminal. Be sure to first set up the hyper parameters from the model. 
