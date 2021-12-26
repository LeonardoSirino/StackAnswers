#%% import
from pprint import pprint

import numpy as np
from gym import Env
from gym.spaces import Box
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy


class Custom_Env(Env):
    def __init__(self):

        # Define the state space

        # State variables
        self.state_1 = 0
        self.state_2 = 0
        self.state_3 = 0
        self.state_4_currentTimeSlots = 0

        # Define the gym components
        self.action_space = Box(
            low=np.array([0, 0, 0]), high=np.array([10, 20, 27]), dtype=np.int
        )

        self.observation_space = Box(
            low=np.array([20, -20, 0, 0]),
            high=np.array([22, 250, 100, 287]),
            dtype=np.float16,
        )

    def step(self, action):

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

        state = np.array(
            [self.state_1, self.state_2, self.state_3, self.state_4_currentTimeSlots]
        )

        # Return step information
        return state, reward, done, info

    def render(self):
        pass

    def reset(self):
        self.state_1 = 0
        self.state_2 = 0
        self.state_3 = 0
        self.state_4_currentTimeSlots = 0
        state = np.array(
            [self.state_1, self.state_2, self.state_3, self.state_4_currentTimeSlots]
        )
        return state


from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam


def build_model(states, actions):
    pprint(f"actions: {actions}")
    pprint(f"states: {states}")

    model = models.Sequential(
        [
            keras.Input(shape=states),
            layers.Dense(24, activation="relu"),
            layers.Dense(24, activation="relu"),
            layers.Dense(actions[0], activation="linear"),
            layers.Reshape(target_shape=(3, )),
        ]
    )
    return model


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(
        model=model,
        memory=memory,
        policy=policy,
        nb_actions=actions,
        nb_steps_warmup=10,
        target_model_update=1e-2,
    )
    return dqn


if __name__ == "__main__":
    #%% Set up the environment
    env = Custom_Env()

    states = env.observation_space.shape
    actions = env.action_space.shape
    # print("env.observation_space: ", env.observation_space)
    # print("env.observation_space.shape : ", env.observation_space.shape)
    # print("action_space: ", env.action_space)
    # print("action_space.shape : ", env.action_space.shape)

    model = build_model(states, actions)
    print(model.summary())

    dqn = build_agent(model, actions)
    dqn.compile(Adam(lr=1e-3), metrics=["mae"])
    dqn.fit(env, nb_steps=4000, visualize=False, verbose=1)
