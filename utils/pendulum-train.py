import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import random


cat = tfp.distributions.Categorical
env = gym.make("Pendulum-v1", render_mode='human')
observation = env.reset()
render = False
reward_sum = 0
rewards = []
pred_actions = []
pred_values = []
taken_action = []
inputs = []
running_reward = None
start = True
batch_size = 64
T = {}

observation = observation[0]
observation[2] = observation[2]/ 8 
observation.shape

def action_loss(rewards, pred_values ):
    
    return loss

huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
critics_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

input_x = tf.keras.Input(shape=(6))
x = tf.keras.layers.Dense(256, activation = "relu")(input_x)
x = tf.keras.layers.Dense(256, activation = "relu")(x)
x = tf.keras.layers.Dense(15)(x)

actor = tf.keras.Model(inputs=input_x, outputs=x)

state_input = tf.keras.Input(shape=(6))
state_x = tf.keras.layers.Dense(256, activation = "relu")(state_input)
last_init = tf.random_uniform_initializer(minval=-7.0, maxval=-3.0)
action_input = tf.keras.Input(shape= 15)
action_x = tf.keras.layers.Dense(256, activation = "relu")(action_input)

x = tf.keras.layers.Concatenate()([state_x, action_x])
x = tf.keras.layers.Dense(128, activation = "relu")(x)
x = tf.keras.layers.Dense(128, activation = "relu")(x)
x = tf.keras.layers.Dense(1, activation = "linear")(x)

critic = tf.keras.Model(inputs=[state_input, action_input], outputs=x)

actor.summary(), critic.summary()


episode = 0 
while True:
    
    if render: env.render()

    if start: 
        print("yess")
        prev_obj = np.zeros((1, 3))
        start = False

    T[episode] = {}
    x =  np.concatenate([observation[np.newaxis, ...], prev_obj], axis=-1)
  
    T[episode]['s'] = x

    action_prob_unnorm = actor(x, training=True)
    action_prob = tf.nn.softmax(action_prob_unnorm)
    # print("this is the highest function", tf.argmax(action_prob, axis=-1).numpy(), action_prob.numpy())
    action = tf.random.categorical(action_prob, 1)[0, 0]
    control_value = controls[action]
    
    
    
    state_value = critic([x, action_prob], training=True)
    
    T[episode]['a'] = action_prob
    T[episode]['s_v'] = state_value

    prev_obj = observation[np.newaxis, ...]
    
    observation, reward, done, truncated, info = env.step([control_value])

    reward_sum += reward
    T[episode]['r'] = reward
    observation[2] = observation[2]/ 8
    observation = observation
    
   
    episode += 1

    if truncated:

        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        f = open(f"rewards.txt", "a+")
        f.write(str(running_reward)+"\n")
        f.close()

        print('resetting env. episode reward total was %f. running mean: %f, avg %f' % (reward_sum, running_reward, reward_sum/200))

        # actor_gradients = []
        # state_gradients = []
        loss_total = 0
        actor_loss_total = 0

        y_hat = []
        states = []
        actions = []
        for i in random.sample(range(0, 199), batch_size):
            y = T[i]['r'] + 0.9 * T[i+1]['s_v']
            y_hat.append(tf.squeeze(y))
            states.append(tf.squeeze(T[i]['s']))
            actions.append(tf.squeeze(T[i]['a']))
        y_hat = np.array(y_hat)[...,tf.newaxis]
        states = np.array(states)
        actions = np.array(actions)

     

        with tf.GradientTape() as state_tape:     
            output_ = critic([states, actions], training=True)
            critic_loss = tf.reduce_mean((y_hat - output_)**2)
        loss_total += critic_loss
        

        with tf.GradientTape() as actor_tape:
            actor_out = actor(states, training=True)
            action_prob = tf.nn.softmax(actor_out)
            output_ = critic([states, action_prob], training=True)
            actor_loss = -tf.reduce_mean(output_)

        actor_loss_total += actor_loss
        state_grad = state_tape.gradient(critic_loss, critic.trainable_weights)
        critics_optimizer.apply_gradients(zip(state_grad, critic.trainable_weights))
        actor_gradient = actor_tape.gradient(actor_loss, actor.trainable_weights)
        actor_optimizer.apply_gradients(zip(actor_gradient, actor.trainable_weights))

        print("Critc Loss: ", loss_total.numpy(), "Actor loss", actor_loss_total.numpy())
        f = open(f"loss.txt", "a+")
        f.write(str(loss_total.numpy())+"\n")
        f.close()
        # for i in range(batch_size):
            
            
                    
        episode = 0
        T = {}
        reward_sum = 0
        observation = env.reset()
        observation = observation[0]
        observation[2] = observation[2]/ 8
       
        #     with tf.GradientTape() as state_tape:

        #         y = T[i]['r'] + 0.9 * T[i+1]['s_v']
                
        #         output_ = critic([T[i]['s'], T[i]['a']], training=True)
        #         critic_loss = tf.sqrt((y - output_)**2)
        #     loss_total += critic_loss
        #     state_gradients.append(state_tape.gradient(critic_loss, critic.trainable_weights))

        #     with tf.GradientTape() as actor_tape:
        #         actor_out = actor(T[i]['s'], training=True)
        #         action_prob = tf.nn.softmax(actor_out)
        #         output_ = critic([T[i]['s'], action_prob], training=True)
        #         actor_loss = -tf.sqrt(output_**2)
                
          
        #     actor_gradients.append(actor_tape.gradient(actor_loss, actor.trainable_weights))
        # f = open(f"loss.txt", "a+")
        # f.write(str(loss_total.numpy()[0,0])+"\n")
        # f.close()
        # for i in range(batch_size):
        #     actor_optimizer.apply_gradients(zip(actor_gradients[i], actor.trainable_weights))
        #     critics_optimizer.apply_gradients(zip(state_gradients[i], critic.trainable_weights))
                    
        # episode = 0
        # T = {}
        # reward_sum = 0
        # observation = env.reset()
        # observation = observation[0]
        # observation[2] = observation[2]/ 8
