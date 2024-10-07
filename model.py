import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras import layers # type: ignore
import numpy as np
from collections import deque
import random

"""
This will be a Q-Network model to hopefully play the Snake game well..
Firstly, I need to define a basic NN that takes an input (Snake environment) and predicts the Q values.
 Input will be a feature vector with snake head (x, y) food (x, y) direction (up, left, down, right)

"""


class QNetwork:
    def __init__(self, state_size=11, action_size=3):
        #Define hyperparameters
        self.epsilon = 1
        self.epsilon_decay = 0.005
        self.epsilon_min = 0.01
        self.gamma = 0.92
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.002

        self.model = self.build_model()

        self.memory = deque(maxlen=2000)


    def build_model(self):
        model = Sequential([
            layers.Dense(64, input_dim=self.state_size, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    """
    For the memory replay, each epsiode will be stored in a tuple using deque. 
    The tuple will be (state, action, reward, next_state, done)
    During training you will sample a batch of experiences from this buffer to update the network
    and adding to the memory will happen after every iteration of the game loop
    """
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    """
    This replay method will be how we sample a mini-batch from the memory buffer during training
    First, check to make sure there are enough experinces in memory. Second, sample a random mini-batch from the experiences
    Use that tuple (with state, reward ....) to equate the bellman equation which will calculate the target Q-value
    Update the Q-value for the chosen action
    Trains the model
    Decay epsilon 
    """
    def replay(self, batch_size):

        #Random sample of mini_batch
        mini_batch = random.sample(self.memory, batch_size) 

        states = np.zeros((batch_size, self.state_size))
        targets = np.zeros((batch_size, self.action_size))

        next_states = np.zeros((batch_size, self.state_size))

        for i in range(batch_size):
            next_states[i] = mini_batch[i][3]

        next_q_values = self.model.predict(next_states)

        for i in range(batch_size):
            state, action, reward, next_state, done = mini_batch[i]
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(next_q_values[i])

            # Current Q-values
            states[i] = state
            target_f = self.model.predict(state.reshape(1, self.state_size))[0]
            target_f[action] = target
            targets[i] = target_f

        # Train the model on all samples at once
        self.model.fit(states, targets, epochs=1, verbose=0)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= (1 - self.epsilon_decay)

        # for state, action, reward, next_state, done in mini_batch:
        #     target = reward
        #     if not done:
        #         # Bellman equation
        #         target = reward + self.gamma * np.amax(self.model.predict(np.reshape(next_state, [1, self.state_size]))[0])

        #     # Update the Q-value for the chosen action
        #     # Target_f holds the predicted Q-value for the current state in a NP array
        #     # 1D np array with len = num_possible_actions (4)
        #     target_f = self.model.predict(np.reshape(state, [1, self.state_size]))
        #     target_f[0][action] = target

        #     #Train the model
        #     self.model.fit(np.reshape(state, [1, self.state_size]), np.reshape(target_f, [1, self.action_size]), epochs=1, verbose=0)

        # #Decay epsilon
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= (1 - self.epsilon_decay)


    """
    This will use the e-greedy policy
    whenever np.randomn.rand() (a random number between 0 and 1) is less than epsilon (it will be during the first epsiode for sure),
    we want to return a random action so that it explores randomly.
    Whenever it is not less than epsilon, we want to use the model to predict the Q-value and return the greatest Q-value
    
    """
    def action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        else:
            q_values = self.model.predict(np.reshape(state, [1, self.state_size]))
            print(q_values)
            return np.argmax(q_values)
        
