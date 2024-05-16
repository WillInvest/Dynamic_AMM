"""
Agent.py

Tested with:
Tensorflow 2.2

Zheng Xing <zxing@stevens.edu>
"""

from memory import *
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate

GAMMA = 0.99
TAU = 0.001
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.0001

STATE_SHAPE = (4,)
ACTION_SHAPE = (1,)
ACTION_LOWER_BOUND = -1.0
ACTION_UPPER_BOUND = 1.0
# ACTIONS = [(ACTION_LOWER_BOUND, ACTION_UPPER_BOUND)]

LOG_DIR = "tensorboard_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

SAVING_RAM = False


class DeepDeterministicPolicyGradient:
    def __init__(self, environment, wandb):
        self.env = environment
        self.wandb = wandb
        self.actor = Actor()
        self.critic = Critic()
        if SAVING_RAM:
            self.memory = Memory()
        else:
            self.memory = Buffer(STATE_SHAPE[0], ACTION_SHAPE[0])
        self.noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(0.01) * np.ones(1))
        self.actor.target_network.set_weights(self.actor.online_network.get_weights())
        self.critic.target_network.set_weights(self.critic.online_network.get_weights())
        self._fill_the_first_piece_of_memory()

    def reset(self):
        state, _ = self.env.reset()
        return state

    @tf.function
    def _update_online_network(self, states_batch, actions_batch, rewards_batch, next_states_batch):
        """
        Update the online network of the Actor and the online network of the Critic.
        :return:
        """
        # TODO: Find a better way to handle next_state when done is True.
        # Update the Critic's online network.
        with tf.GradientTape() as tape:
            target_actions = self.actor.target_network(next_states_batch, training=True)
            targets = rewards_batch + GAMMA * self.critic.target_network([next_states_batch, target_actions],
                                                                         training=True)
            action_values = self.critic.online_network([states_batch, actions_batch], training=True)
            loss = tf.math.reduce_mean(tf.math.square(targets - action_values))
        gradients = tape.gradient(loss, self.critic.online_network.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(gradients, self.critic.online_network.trainable_variables))

        # Update the Actor's online network.
        with tf.GradientTape() as tape:
            online_actions = self.actor.online_network(states_batch, training=True)
            action_values = self.critic.online_network([states_batch, online_actions], training=True)
            objective = tf.math.reduce_mean(action_values)
            neg_objective = - objective
        gradients = tape.gradient(neg_objective, self.actor.online_network.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(gradients, self.actor.online_network.trainable_variables))

        return loss, neg_objective

    def update_online_network(self, states_batch, actions_batch, rewards_batch, next_states_batch):
        loss, neg_objective = self._update_online_network(states_batch, actions_batch, rewards_batch, next_states_batch)

        # Log outside the tf.function to ensure the tensors are converted to numpy arrays
        self.wandb.log({"critic_loss": loss.numpy(), "actor_objective": -neg_objective.numpy()})

    def policy(self, state, no_noise=False):
        network_input = np.expand_dims(state, axis=0)
        network_output = self.actor.online_network(network_input)
        action = tf.squeeze(network_output)

        if not no_noise:
            action = action + self.noise()
            action = np.clip(action, -0.999, 0.999)

        if np.ndim(action) > 0:
            action = action[0]
        self.wandb.log({"actions": action, "noise": self.noise() if not no_noise else 0})
        return action

    @tf.function
    def update_target_network(self, tau=TAU):
        """
        Implementing the 'soft' target update. This is supposed to be called everytime the online networks are updated.
        Since the Tau is far less than 1, the target networks are updated towards the online networks slower. The goal
        is to make convergence more stable.
        :param tau: tau should be far less than 1.
        :return:
        """
        critic_target_weights = self.critic.target_network.variables
        critic_online_weights = self.critic.online_network.variables
        for (c_t_w, c_o_w) in zip(critic_target_weights, critic_online_weights):
            c_t_w.assign(c_o_w * tau + c_t_w * (1 - tau))

        actor_target_weights = self.actor.target_network.variables
        actor_online_weights = self.actor.online_network.variables
        for (a_t_w, a_o_w) in zip(actor_target_weights, actor_online_weights):
            a_t_w.assign(a_o_w * tau + a_t_w * (1 - tau))

    def _fill_the_first_piece_of_memory(self):
        next_state = self.reset()
        done = False
        for i in range(REPLAY_BATCH_SIZE):
            if done:
                next_state = self.reset()
            state = np.copy(next_state)
            action = self.policy(state)
            next_state, reward, done, truncated, _ = self.env.step(action)
            if SAVING_RAM:
                experience = (state, action, reward, done)
            else:
                experience = (state, action, reward, next_state, done)
            self.memory.add(experience)


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.x_prev = x_initial
        self.reset()

    def __call__(self):
        x = (
                self.x_prev
                + self.theta * (self.mean - self.x_prev) * self.dt
                + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Actor:
    def __init__(self):
        self.state = np.ndarray(STATE_SHAPE)
        # self.actions = ACTIONS
        self.online_network = self._build_network(name='Actor_Online_Network')
        self.target_network = self._build_network(name='Actor_Target_Network')
        # self.online_network.summary()
        # self.target_network.summary()

        self.optimizer = tf.optimizers.Adam(ACTOR_LEARNING_RATE)

    """
    Note: We need the initialization for last layer of the Actor to be between
    `-0.003` and `0.003` as this prevents us from getting `1` or `-1` output values in
    the initial stages, which would squash our gradients to zero, as we use the `tanh` activation.
    """

    def _build_network(self, name=''):
        network_input = Input(shape=self.state.shape)
        fc_1 = Dense(256, activation=tf.keras.activations.relu,
                     kernel_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003))(network_input)
        fc_2 = Dense(256, activation=tf.keras.activations.relu,
                     kernel_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003))(fc_1)
        action = Dense(ACTION_SHAPE[0],
                       activation=tf.keras.activations.tanh,
                       kernel_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
                       )(fc_2)
        scaled_action = action * ACTION_UPPER_BOUND
        network = tf.keras.Model(inputs=network_input, outputs=scaled_action, name=name)
        # tf.keras.utils.plot_model(network, name, show_shapes=True)
        return network


class Critic:
    def __init__(self):
        self.state = np.ndarray(STATE_SHAPE)
        # self.actions = ACTIONS
        self.online_network = self._build_network(name='Critic_Online_Network')
        self.target_network = self._build_network(name='Critic_Target_Network')
        # self.online_network.summary()
        # self.target_network.summary()

        self.optimizer = tf.optimizers.Adam(CRITIC_LEARNING_RATE)

    def _build_network(self, name=''):
        state_input = Input(shape=self.state.shape)
        state_output = Dense(16, activation=tf.keras.activations.relu)(state_input)
        state_output = Dense(32, activation=tf.keras.activations.relu)(state_output)

        action_input = Input(shape=ACTION_SHAPE)
        action_output = Dense(32, activation=tf.keras.activations.relu)(action_input)

        concatenation = Concatenate()([state_output, action_output])

        fc = Dense(256, activation=tf.keras.activations.relu,
                   kernel_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
                   )(concatenation)
        fc = Dense(256, activation=tf.keras.activations.relu,
                   kernel_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
                   )(fc)
        output = Dense(1, activation=None)(fc)

        network = tf.keras.Model(inputs=[state_input, action_input], outputs=output)
        # tf.keras.utils.plot_model(network, name, show_shapes=True)

        return network


def normalize_rewards(reward_list):
    mean_reward = np.mean(reward_list)
    std_reward = np.std(reward_list)
    normalized_rewards = [(r - mean_reward) / std_reward for r in reward_list]
    return normalized_rewards
