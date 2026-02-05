
import numpy as np
import random
from sklearn.neural_network import MLPRegressor
from collections import deque, defaultdict

class DQLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        
        self.gamma = 0.95        # discount rate
        self.epsilon = 1.0       # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Scikit-learn MLPRegressor for Q-learning
        self.model = MLPRegressor(hidden_layer_sizes=(24, 24), activation='relu', 
                                  solver='adam', learning_rate_init=self.learning_rate,
                                  max_iter=1, warm_start=True)
        
        # Initialize model with random data to set up dimensions
        # This is a bit of a hack because partial_fit needs to know the shape
        X_init = np.zeros((1, state_size))
        y_init = np.zeros((1, action_size))
        self.model.fit(X_init, y_init)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict([state])
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        
        states = np.array([i[0] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        
        # Batch prediction for efficiency
        current_qs_list = self.model.predict(states)
        next_qs_list = self.model.predict(next_states)
        
        X_train = []
        y_train = []
        
        for index, (state, action, reward, next_state, done) in enumerate(minibatch):
            current_qs = current_qs_list[index]
            
            if done:
                target_reward = reward
            else:
                target_reward = reward + self.gamma * np.amax(next_qs_list[index])
            
            current_qs[action] = target_reward
            
            X_train.append(state)
            y_train.append(current_qs)
            
        # Update model
        self.model.partial_fit(X_train, y_train)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class QLAgent:
    def __init__(self, state_size, action_size):
        self.action_size = action_size
        self.q_table = defaultdict(float) # Maps (state_tuple, action) -> Q-value
        
        self.learning_rate = 0.1
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999 # Decay slower for tabular
        
    def discretize(self, state):
        # State: [speed, obstacle_x, obstacle_width, obstacle_height]
        speed = int(state[0] / 2) # buckets of size 2
        obs_x = int(state[1] / 50) # buckets of size 50
        obs_w = int(state[2] / 10) # buckets of size 10
        obs_h = int(state[3] / 10) # buckets of size 10
        return (speed, obs_x, obs_w, obs_h)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_key = self.discretize(state)
        q_values = [self.q_table[(state_key, a)] for a in range(self.action_size)]
        return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        state_key = self.discretize(state)
        next_state_key = self.discretize(next_state)
        
        current_q = self.q_table[(state_key, action)]
        
        if done:
            target = reward
        else:
            next_max_q = max([self.q_table[(next_state_key, a)] for a in range(self.action_size)])
            target = reward + self.gamma * next_max_q
            
        # Q-learning update rule
        self.q_table[(state_key, action)] += self.learning_rate * (target - current_q)
        
        # Decay epsilon here since we don't have a replay batch step
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay(self, batch_size):
        pass # No exp replay for tabular Q-learning
