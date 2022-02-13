#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 05:36:04 2022

Neural network RL policies for cart-pole game.

@author: shauno
"""

#Estimates probability of each action and then selects an action
# randomly based on the probabilities
import tensorflow as tf
from tensorflow import keras
import numpy as np
import gym

def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)
    plt.close()
    return anim

def render_policy_net(model, n_max_steps=200, seed=42):
    frames = []
    env = gym.make("CartPole-v1")
    env.seed(seed)
    np.random.seed(seed)
    obs = env.reset()
    for step in range(n_max_steps):
        frames.append(env.render(mode="rgb_array"))
        left_proba = model.predict(obs.reshape(1, -1))
        action = int(np.random.rand() > left_proba)
        obs, reward, done, info = env.step(action)
        if done:
            break
    env.close()
    return frames

N_INPUTS = 4 # sizee of observation array: env.observation_space.shape[0]

model = keras.models.Sequential([
    keras.layers.Dense(5,activation='elu', input_shape=[N_INPUTS]),
    keras.layers.Dense(1, activation='sigmoid'),
])

#policy gradient (PG) algorithms:
# One PG algorithm is REINFORCE (Williams, 1992)
# 1) NN policy plays game several times, compute gradients
# 2) After several episodes, compute each action's advantage
#     compared to other actions
# 3) Multiply gradient vector by action's advantage
# 4) compute mean of all gradient vectors and perform gradient descent step

def play_one_step(env, obs, model, loss_fn):
    '''
    Parameters
    env : Environment object that the function will interact with.
    obs : Observations resulting from previous action
    model : RL model
    loss_fn : Tloss function used to optimize model

    Returns
    obs : Observation after current step
    reward : rewards received from action
    done : whether the end condition has been met
    grads : gradients for the action taken
    '''
    with tf.GradientTape() as tape:
        left_proba = model(obs[np.newaxis])
        action = (tf.random.uniform([1,1]) > left_proba)
        y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32)
        loss = tf.reduce_mean(loss_fn(y_target, left_proba))
    grads = tape.gradient(loss, model.trainable_variables)
    obs, reward, done, info = env.step(int(action[0,0].numpy()))
    return obs, reward, done, grads

def play_multiple_episodes(env, n_episodes, n_max_steps, model, loss_fn):
    '''
    Parameters
    env : Environment object that the function will interact with.
    n_episodes: Number of episodes to play
    n_max_steps : max number of steps per episode
    model: RL model
    loss_fn : Tloss function used to optimize model

    Returns
    all_rewards : vector of rewards from each step
    all_grads : vector of gradients from eachs tep
    '''
    all_rewards = []
    all_grads = []
    for episode in range(n_episodes):
        current_rewards = []
        current_grads = []
        obs = env.reset()
        for step in range(n_max_steps):
            obs, reward, done, grads = play_one_step(env, obs, model, loss_fn)
            current_rewards.append(reward)
            current_grads.append(grads)
            if done:
                break
        all_rewards.append(current_rewards)
        all_grads.append(current_grads)
        return all_rewards, all_grads

def discount_rewards(rewards, discount_factor):
    '''
    Parameters
    rewards : vector of rewards
    discount_factor: gamma param to specify the value of actions

    Returns
    discounted: vector of discounted rewards
    '''
    discounted = np.array(rewards)
    for step in range(len(rewards)-2, -1, -1):
        discounted[step] += discounted[step + 1] * discount_factor
    return discounted

def discount_and_normalize_rewards(all_rewards, discount_factor):
    '''
    Parameters
    all_rewards : vector of rewards
    discount_factor: gamma param to specify the value of actions

    Returns
    normalized multi dimensional array of rewards
    '''
    all_discounted_rewards = [discount_rewards(rewards, discount_factor)
                              for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean) / reward_std
            for discounted_rewards in all_discounted_rewards]

# test it out:
#discount_rewards([10,0,-50], discount_factor=0.8)
#discount_and_normalize_rewards([[10,0,-50],[10,20]], discount_factor=0.8)

# model hyperparameters:
N_ITERATIONS = 150
N_EPISODES_PER_UPDATE = 10
N_MAX_STEPS = 200
DISCOUNT_FACTOR = 0.95 # actions 13 steps into the future are worth half as much as immediate rewards

OPTIMIZER = keras.optimizers.Adam(lr=0.01)
LOSS_FN = keras.losses.binary_crossentropy

# To plot pretty figures
#%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# To get smooth animations
import matplotlib.animation as animation
mpl.rc('animation', html='jshtml')

env = gym.make('CartPole-v1')

env.seed(42)

for iteration in range(N_ITERATIONS):
    all_rewards, all_grads = play_multiple_episodes(
        env, N_EPISODES_PER_UPDATE, N_MAX_STEPS, model, LOSS_FN)
    all_final_rewards = discount_and_normalize_rewards(all_rewards, DISCOUNT_FACTOR)
    all_mean_grads = []
    for var_index in range(len(model.trainable_variables)):
        mean_grads = tf.reduce_mean(
            [final_reward * all_grads[episode_index][step][var_index]
             for episode_index, final_rewards in enumerate(all_final_rewards)
             for step, final_reward in enumerate(final_rewards)], axis=0)
        all_mean_grads.append(mean_grads)
    OPTIMIZER.apply_gradients(zip(all_mean_grads, model.trainable_variables))

frames = render_policy_net(model)
plot_animation(frames)