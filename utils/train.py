import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import random
import pandas as pd

from model.rl_model import Actor, Critic 

actor = Actor()
critic = Critic()

target_actor = Actor()
target_critic = Critic()

env = gym.make("DemonAttack-v0", render_mode='human')

observation = env.reset()
render = True
start = True
batch_size = 512
reward_sum = 0
running_reward = None
initial_run = False

observation = observation[0]

for k in range(5000):
    for t in range(1000):
        T = {}
        T['s'] = observation
        T['p_s'] = prev 
        prev = observation
        
        noise = np.random.normal(loc=0.0, scale=.2, size=1)
        
        if initial_run == True:
            action = env.action_space.sample()
            observation, reward, done, truncated, info = env.step(action)
            
        else:
            prep_inp = np.concatenate([T["s"], T["p_s"]], axis=-1)[np.newaxis,...]/255.
            action = actor(prep_inp)
            action = action.numpy() + noise
            action = action * 5
            action = np.clip(action, 0, 5)
            action = round(np.squeeze(action).item())
            observation, reward, done, truncated, info = env.step(action)
        
        T["s_1"] = observation
        T['a'] = action
        T['r'] = reward
        T['d'] = done
        

       
        
        trajectories.append(T)
        
        if done:
            observation = env.reset()
            observation = observation[0]
        
    initial_run = False
    
    discounted_rewards, new_tag, count = return_discounted_rewards(trajectories)

    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards)

    # step = 0
    critic_loss_total = 0
    actor_loss_total = 0
    # for i_ in range(1000):

    for i_ in range(1000):
        mini_batch_sample = np.random.choice(range(len(new_tag)), 8)
        
        t_reward = []
        current_states = []
        next_states = []
        state_actions = []
        dones = []
        for i in mini_batch_sample:
            t_reward.append(discounted_rewards[i])
            current_states.append(np.concatenate([new_tag[i]["s"], new_tag[i]["p_s"]], axis=-1)/255.)
            next_states.append(np.concatenate([new_tag[i]["s_1"], new_tag[i]["s"]], axis=-1)/255)
            state_actions.append(new_tag[i]["a"])
            dones.append(new_tag[i]["d"])
    
        current_states = np.array(current_states)
        next_states = np.array(next_states)
        t_reward = np.array(t_reward)[..., np.newaxis]
        state_actions = np.array(state_actions)/5.
        dones = np.array(dones)[..., np.newaxis]
    
        with tf.GradientTape() as critic_tape:
            target_actions = target_actor(next_states, training=True)
            y = t_reward + 0.9 * (1 - dones)* target_critic([next_states, target_actions], training=True)
            critic_value = critic([current_states, state_actions], training=True)  
            
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
        critic_loss_total += critic_loss
    
        state_grad = critic_tape.gradient(critic_loss, critic.trainable_weights)
        critics_optimizer.apply_gradients(zip(state_grad, critic.trainable_weights))
    
       
        
        with tf.GradientTape() as actor_tape:
            actor_out = actor(current_states, training=True)
            output_ = critic([current_states, actor_out], training=True)
            actor_loss = -tf.math.reduce_mean(output_)
    
        actor_loss_total += actor_loss
        actor_gradient = actor_tape.gradient(actor_loss, actor.trainable_weights)
        actor_optimizer.apply_gradients(zip(actor_gradient, actor.trainable_weights))
    
    
        temp = []
        for i in range(len(critic.get_weights())):
            calculate_weight = critic.get_weights()[i] * 0.005  + target_critic.get_weights()[i] *  (1-0.005)
            temp.append(calculate_weight)
        target_critic.set_weights(temp)
        
        temp = []
        for i in range(len(actor.get_weights())):
            calculate_weight = actor.get_weights()[i] * 0.005 + target_actor.get_weights()[i] * (1 - 0.005)
            temp.append(calculate_weight)
        target_actor.set_weights(temp)
        
        print(f"Update: {i_}, actor_error: {actor_loss.numpy()}, critic_error: {critic_loss.numpy()}", end="\r", flush=True)
    
        # rewards_mean = np.mean(rewards_collection[-1000:])
    
    print()
    print("Episode: ", i_, "transitions: ", len(trajectories), "critic_loss: ", critic_loss_total.numpy(), "actor_loss: ", actor_loss_total.numpy())
    # f = open(f"rewards.txt", "a+")
    # f.write(str(rewards_mean)+"\n")
    # f.close()
    
    f = open(f"critic_loss.txt", "a+")
    f.write(str(critic_loss_total.numpy())+"\n")
    f.close()
    f = open(f"actor_loss.txt", "a+")
    f.write(str(actor_loss_total.numpy())+"\n")
    f.close() 

    

    
        
