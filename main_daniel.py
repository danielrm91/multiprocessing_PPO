'''
this file is based on the work on 
https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8
without some functions that are not convenient for the purpose
'''

import gym
import gym_kondili
import sys
import torch
import multiprocessing as mp

from arguments_daniel import get_args
from eval_policy_daniel_iterative2 import eval_policy
from ppo_daniel import PPO                     #import this class from ppo
from network_daniel import Attention_module_actor  #import this class from network
from network_daniel import Attention_module_critic #import this class from network

def train(env, hyperparameters, actor_model, critic_model):
    print(f"Training",flush=True)
    #Create a model for PPO
    model = PPO(policy_class=Attention_module_actor, value_class=Attention_module_critic, env=env, **hyperparameters)
    
    #load existing model
    if actor_model != '' and critic_model != '':
        print(f"Loading in {actor_model} and {critic_model}...", flush=True)
        model.actor.load_state_dict(torch.load(actor_model))
        model.critic.load_state_dict(torch.load(critic_model))
        print(f"Succesfully loadede.",flush=True)
    elif actor_model != '' or critic_model != '':
        print(f"Error: Either specify both actor/critic models or non at all. We don't want to accidentally override anything!")
        sys.exit(0)
    else:
        print(f"Training from scratch.", flush=True)

    num_epochs = 100
    num_workers = 5
    print(f"Running for a total of {num_epochs} timesteps")
    for epoch in range(num_epochs):
        trajectories = model.parallel_rollouts(num_workers)
        model.learn(trajectories, num_workers, num_epochs, curr_epoch=epoch) 

    #model.learn(total_timesteps = 1_674_000) #15_500_000

def test(env, actor_model):
    print(f"Testing {actor_model}", flush=True)
    if actor_model =='':
        print(f"Didn't specify model file. Exiting", flush=True)
        sys.exit(0)
    policy = Attention_module_actor()
    policy.load_state_dict(torch.load(actor_model))
    eval_policy(policy=policy, env=env)
    
def main(args):
    hyperparameters = {
        'timesteps_per_batch':           62, #620,    # this is per worker so if I have 7 workers this is 62*7
        'max_timesteps_per_episode':     31,
        'gamma':                       0.99,
        'n_updates_per_iteration':        6,    
        'lr':                          1e-4,    #1e-4
        'clip':                         0.2,    
        'ent_init':                     0.2,    #0.2
        'max_grad_norm':                0.5,
        'var':                         1e-2,    #1e-2
        'temp':                           1,    #1
        'num_minibatches':                5,
        'lam':                         0.98,
        'target_kl':                   0.02
    }
    
    env = gym.make('kondili-v0')

    if args.mode == 'train':
        train(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model)
    else:
        test(env=env, actor_model=args.actor_model)
        
if __name__ == '__main__':
    args = get_args()
    main(args)