from dataclasses import dataclass
from typing import List

import numpy as np
import tensorflow as tf
from gym import Env
from gym.spaces import Box
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam


@dataclass
class TrainingConfig:
    seed = 42
    gamma = 0.99  # Discount factor for past rewards
    epsilon = 1.0  # Epsilon greedy parameter
    epsilon_min = 0.1  # Minimum epsilon greedy parameter
    epsilon_max = 1.0  # Maximum epsilon greedy parameter
    batch_size = 32  # Size of batch taken from replay buffer
    max_steps_per_episode = 10000

    # Number of frames to take random action and observe output
    epsilon_random_frames = 50000
    # Number of frames for exploration
    epsilon_greedy_frames = 1000000.0
    # Maximum replay length
    # Note: The Deepmind paper suggests 1000000 however this causes memory issues
    max_memory_length = 100000
    # Train the model after 4 actions
    update_after_actions = 4
    # How often to update the target network
    update_target_network = 10000

    def __post_init__(self):
        # Rate at which to reduce chance of random action being taken
        self.epsilon_interval = self.epsilon_max - self.epsilon_min


class Session:
    def __init__(self, config: TrainingConfig,
                 model: keras.Model,
                 target_model: keras.Model,
                 env: Env) -> None:

        self.config = config
        self.model = model
        self.target_model = target_model
        self.env = env

    def train(self) -> None:
        optimizer = keras.optimizers.Adam(learning_rate=1e-3)

        # Experience replay buffers
        action_history = []
        state_history = []
        state_next_history = []
        rewards_history = []
        done_history = []
        episode_reward_history = []
        running_reward = 0
        episode_count = 0
        frame_count = 0

        num_actions = 4

        # Using huber loss for stability
        loss_function = keras.losses.Huber()

        epsilon = self.config.epsilon

        while True:  # Run until solved
            state = self.env.reset()
            episode_reward = 0

            for _ in range(1, self.config.max_steps_per_episode):
                # env.render(); Adding this line would show the attempts
                # of the agent in a pop up window.
                frame_count += 1

                # Use epsilon-greedy for exploration
                if frame_count < self.config.epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                    action = np.random.rand(num_actions)
                else:
                    # Predict action Q-values
                    # From environment state
                    state_tensor = tf.convert_to_tensor(state)
                    state_tensor = tf.expand_dims(state_tensor, 0)
                    action_probs = self.model(state_tensor, training=False)

                    print("action_probs")
                    print(action_probs)
                    # Take best action
                    action = tf.argmax(action_probs[0]).numpy()

                # Decay probability of taking random action
                epsilon -= self.config.epsilon_interval / self.config.epsilon_greedy_frames
                epsilon = max(epsilon, self.config.epsilon_min)

                # Apply the sampled action in our environment
                state_next, reward, done, _ = self.env.step(action)
                state_next = np.array(state_next)

                episode_reward += reward

                # Save actions and states in replay buffer
                action_history.append(action)
                state_history.append(state)
                state_next_history.append(state_next)
                done_history.append(done)
                rewards_history.append(reward)
                state = state_next

                # Update every fourth frame and once batch size is over 32
                if frame_count % self.config.update_after_actions == 0 and len(done_history) > self.config.batch_size:

                    # Get indices of samples for replay buffers
                    indices = np.random.choice(
                        range(len(done_history)), size=self.config.batch_size)

                    # Using list comprehension to sample from replay buffer
                    state_sample = np.array(
                        [state_history[i] for i in indices])
                    state_next_sample = np.array(
                        [state_next_history[i] for i in indices])
                    rewards_sample = [rewards_history[i] for i in indices]
                    action_sample = [action_history[i] for i in indices]
                    done_sample = tf.convert_to_tensor(
                        [float(done_history[i]) for i in indices]
                    )

                    # Build the updated Q-values for the sampled future states
                    # Use the target model for stability
                    future_rewards = self.target_model.predict(
                        state_next_sample)
                    # Q value = reward + discount factor * expected future reward
                    updated_q_values = rewards_sample + self.config.gamma * tf.reduce_max(
                        future_rewards, axis=1
                    )

                    # If final frame set the last value to -1
                    updated_q_values = updated_q_values * \
                        (1 - done_sample) - done_sample

                    # Create a mask so we only calculate loss on the updated Q-values
                    masks = tf.one_hot(action_sample, num_actions)

                    with tf.GradientTape() as tape:
                        # Train the model on the states and updated Q-values
                        q_values = self.model(state_sample)

                        # Apply the masks to the Q-values to get the Q-value for action taken
                        q_action = tf.reduce_sum(
                            tf.multiply(q_values, masks), axis=1)
                        # Calculate loss between new Q-value and old Q-value
                        loss = loss_function(updated_q_values, q_action)

                    # Backpropagation
                    grads = tape.gradient(loss, self.model.trainable_variables)
                    optimizer.apply_gradients(
                        zip(grads, self.model.trainable_variables))

                if frame_count % self.config.update_target_network == 0:
                    # update the the target network with new weights
                    self.model_target.set_weights(self.model.get_weights())
                    # Log details
                    template = "running reward: {:.2f} at episode {}, frame count {}"
                    print(template.format(running_reward,
                          episode_count, frame_count))

                # Limit the state and reward history
                if len(rewards_history) > self.config.max_memory_length:
                    del rewards_history[:1]
                    del state_history[:1]
                    del state_next_history[:1]
                    del action_history[:1]
                    del done_history[:1]

                if done:
                    break

            # Update running reward to check condition for solving
            episode_reward_history.append(episode_reward)
            if len(episode_reward_history) > 100:
                del episode_reward_history[:1]
            running_reward = np.mean(episode_reward_history)

            episode_count += 1

            if running_reward > 40:  # Condition to consider the task solved
                print("Solved at episode {}!".format(episode_count))
                break


class Custom_Env(Env):
    def __init__(self):
        self.state_1 = 0
        self.state_2 = 0
        self.state_3 = 0
        self.state_4_currentTimeSlots = 0

        # Define the gym components
        self.action_space = Box(low=np.array(
            [0, 0, 0]), high=np.array([10, 20, 27]), dtype=np.int)

        self.observation_space = Box(low=np.array(
            [20, -20, 0, 0]), high=np.array([22, 250, 100, 287]), dtype=np.float16)

    def step(self, action: List[float]):
        print("Action: ", action)
        # Update state variables
        self.state_1 = self.state_1 + action[0]
        self.state_2 = self.state_2 + action[1]
        self.state_3 = self.state_3 + action[2]

        # Calculate reward
        reward = self.state_1 + self.state_2 + self.state_3

        # Set placeholder for info
        info = {}

        # Check if it's the end of the day
        if self.state_4_currentTimeSlots >= 287:
            done = True
        if self.state_4_currentTimeSlots < 287:
            done = False

        # Move to the next timeslot
        self.state_4_currentTimeSlots += 1

        state = np.array([
            self.state_1,
            self.state_2,
            self.state_3,
            self.state_4_currentTimeSlots
        ])

        # Return step information
        return state, reward, done, info

    def render(self):
        pass

    def reset(self):
        self.state_1 = 0
        self.state_2 = 0
        self.state_3 = 0
        self.state_4_currentTimeSlots = 0
        state = np.array([
            self.state_1,
            self.state_2,
            self.state_3,
            self.state_4_currentTimeSlots,
        ])

        return state


def build_model(num_states, num_actions: int) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Dense(24, input_shape=num_states, activation='relu'),
            layers.Dense(24, activation='relu'),
            layers.Dense(num_actions, activation='linear')
        ]
    )

    return model


def main():
    env = Custom_Env()

    config = TrainingConfig()

    num_states = env.observation_space.shape
    num_actions = env.action_space.shape[0]

    model = build_model(num_states, num_actions)
    target_model = build_model(num_states, num_actions)

    print(model.summary())

    session = Session(config, model, target_model, env)
    session.train()


if __name__ == "__main__":
    main()
