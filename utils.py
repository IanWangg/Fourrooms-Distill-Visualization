#@title Training Methods
from tqdm import tqdm

import gym
import copy

import numpy as np
import torch

from buffer import ReplayBuffer


def eval_policy(policy, env, eval_episodes=10):
    eval_env = env

    avg_reward = 0
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state), eval=True)
            state, reward, done, info = eval_env.step(action)
            done_float = float(done)
            avg_reward += 1

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward
            
def train_agent(
    agent,
    env,
    max_steps=int(5e4),
    replay_buffer_size=int(1e5),
    exploration_steps=int(1e3),
    eval_frequency=int(1e3),
    replay_buffer=None,
    batch_size=16,
):
    eval_env = copy.deepcopy(env)

    if replay_buffer is None:
        replay_buffer = ReplayBuffer(max_size=replay_buffer_size)

    evaluations = []

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    policy = agent
    
    # start training the agent
    for t in range(int(max_steps)):
        episode_timesteps += 1
        
        if t < exploration_steps:
            action = env.action_space.sample()
        else:
            action = policy.select_action(np.array(state), eval=True)

        # Perform action and log results
        next_state, reward, done, info = env.step(action)
        episode_reward += reward

        # determine if the agent reaches the goal
        done_float = float(done)
            
        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_float)
        state = copy.deepcopy(next_state)

        if t >= exploration_steps:
            policy.train(replay_buffer, batch_size)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            # print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            

        # Evaluate episode
        if (t + 1) % eval_frequency == 0:
            print(f'Evaluation at time step : {t+1}')
            evaluations.append(eval_policy(policy, env))

    return policy, evaluations

