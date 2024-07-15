import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers

import tensorflow as tf
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

"""
We use [Gymnasium](https://gymnasium.farama.org/) to create the environment.
We will use the `upper_bound` parameter to scale our actions later.
"""

# Specify the `render_mode` parameter to show the attempts of the agent in a pop up window.
env = gym.make("Pendulum-v1")  # , render_mode="human")

num_states = env.observation_space.shape[0]
print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))


"""
The `Buffer` class implements Experience Replay.

---
![Algorithm](https://i.imgur.com/mS6iGyJ.jpg)
---


**Critic loss** - Mean Squared Error of `y - Q(s, a)`
where `y` is the expected return as seen by the Target network,
and `Q(s, a)` is action value predicted by the Critic network. `y` is a moving target
that the critic model tries to achieve; we make this target
stable by updating the Target model slowly.

**Actor loss** - This is computed using the mean of the value given by the Critic network
for the actions taken by the Actor network. We seek to maximize this quantity.

Hence we update the Actor network so that it produces actions that get
the maximum predicted value as seen by the Critic, for a given state.
"""


class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0

        self.state_buffer = np.zeros(
            (self.buffer_capacity, num_states), dtype=np.float32
        )
        self.action_buffer = np.zeros(
            (self.buffer_capacity, num_actions), dtype=np.float32
        )
        self.reward_buffer = np.zeros((self.buffer_capacity, 1), dtype=np.float64)
        self.next_state_buffer = np.zeros(
            (self.buffer_capacity, num_states), dtype=np.float32
        )

    def record(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    @tf.function
    def update(
        self,
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
    ):
        with tf.GradientTape() as tape:
            target_q_values = target_actor(next_state_batch, training=True)
            max_target_q_value = tf.reduce_max(target_q_values, axis=1, keepdims=True)
            y = tf.cast(reward_batch, tf.float32) + gamma * max_target_q_value

            q_values = actor_model(state_batch, training=True)
            q_values = tf.reduce_sum(q_values * action_batch, axis=1, keepdims=True)
            loss = keras.losses.MeanSquaredError()(y, q_values)

        grads = tape.gradient(loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(zip(grads, actor_model.trainable_variables))

    def learn(self):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)

        # print(batch_indices)

        state_batch = tf.convert_to_tensor(
            self.state_buffer[batch_indices], dtype=tf.float32
        )
        action_batch = tf.convert_to_tensor(
            self.action_buffer[batch_indices], dtype=tf.float32
        )
        reward_batch = tf.convert_to_tensor(
            self.reward_buffer[batch_indices], dtype=tf.float64
        )
        next_state_batch = tf.convert_to_tensor(
            self.next_state_buffer[batch_indices], dtype=tf.float32
        )

        # print(state_batch, action_batch, reward_batch, next_state_batch)

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
def update_target(target, original, tau):
    target_weights = target.get_weights()
    original_weights = original.get_weights()

    for i in range(len(target_weights)):
        target_weights[i] = original_weights[i] * tau + target_weights[i] * (1 - tau)

    target.set_weights(target_weights)


"""
Here we define the Actor and Critic networks. These are basic Dense models
with `ReLU` activation.

Note: We need the initialization for last layer of the Actor to be between
`-0.003` and `0.003` as this prevents us from getting `1` or `-1` output values in
the initial stages, which would squash our gradients to zero,
as we use the `tanh` activation.
"""


def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

    # Our upper bound is 2.0 for Pendulum.
    outputs = outputs * upper_bound
    model = keras.Model(inputs, outputs)
    return model


"""
`policy()` returns an action sampled from our Actor network plus some noise for
exploration.
"""


def policy(state, epsilon=0.05):
    if np.random.random() < epsilon:
        return [np.random.uniform(low=lower_bound, high=upper_bound)]
    else:
        sampled_actions = keras.ops.squeeze(actor_model(state))
        sampled_actions = sampled_actions.numpy()
        return [np.squeeze(sampled_actions)]


"""
## Training hyperparameters
"""


actor_model = get_actor()
# critic_model = get_critic()

target_actor = get_actor()
# target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
# target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.01

# critic_optimizer = keras.optimizers.Adam(critic_lr)
actor_optimizer = keras.optimizers.Adam(actor_lr)

total_episodes = 100
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 1.0

buffer = Buffer(50000, 50)

"""
Now we implement our main training loop, and iterate over episodes.
We sample actions using `policy()` and train with `learn()` at each time step,
along with updating the Target networks at a rate `tau`.
"""

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

# Takes about 4 min to train
for ep in range(total_episodes):
    prev_state, _ = env.reset()
    episodic_reward = 0

    while True:
        tf_prev_state = keras.ops.expand_dims(
            keras.ops.convert_to_tensor(prev_state), 0
        )

        action = policy(tf_prev_state)
        # Receive state and reward from environment.
        state, reward, done, truncated, _ = env.step(action)

        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward

        # update_target(target_critic, critic_model, tau)

        # End this episode when `done` or `truncated` is True
        if done or truncated:
            break

        prev_state = state

    buffer.learn()

    if ep % 10 == 0:
        update_target(target_actor, actor_model, tau)

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)

# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Episodic Reward")
plt.show()

"""
If training proceeds correctly, the average episodic reward will increase with time.

Feel free to try different learning rates, `tau` values, and architectures for the
Actor and Critic networks.

The Inverted Pendulum problem has low complexity, but DDPG work great on many other
problems.

Another great environment to try this on is `LunarLander-v2` continuous, but it will take
more episodes to obtain good results.
"""

# Save the weights
actor_model.save_weights("pendulum_actor.weights.h5")
# critic_model.save_weights("pendulum_critic.weights.h5")

target_actor.save_weights("pendulum_target_actor.weights.h5")
# target_critic.save_weights("pendulum_target_critic.weights.h5")

# for ep in range(1):
#     prev_state, _ = env.reset()
#     episodic_reward = 0

#     while True:
#         env.render()
#         tf_prev_state = keras.ops.expand_dims(
#             keras.ops.convert_to_tensor(prev_state), 0
#         )

#         action = policy(tf_prev_state, ou_noise)
#         # Receive state and reward from environment.
#         state, reward, done, truncated, _ = env.step(action)

#         buffer.record((prev_state, action, reward, state))
#         episodic_reward += reward

#         buffer.learn()

#         update_target(target_actor, actor_model, tau)
#         # update_target(target_critic, critic_model, tau)

#         # End this episode when `done` or `truncated` is True
#         if done or truncated:
#             break

#         prev_state = state

#     ep_reward_list.append(episodic_reward)

#     # Mean of last 40 episodes
#     avg_reward = np.mean(ep_reward_list[-40:])
#     print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
#     avg_reward_list.append(avg_reward)
