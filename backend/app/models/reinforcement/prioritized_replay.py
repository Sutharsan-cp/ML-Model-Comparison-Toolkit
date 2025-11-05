import numpy as np
import random
import tensorflow as tf
import heapq

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity)
        self.position = 0
        self.size = 0
    
    def add(self, experience, priority):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size, beta=0.4):
        if self.size == 0:
            return [], [], []
        
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(self.size, batch_size, p=probabilities)
        experiences = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        return experiences, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

class PrioritizedDQN:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.99,
                 exploration_rate=1.0, exploration_decay=0.995, min_exploration=0.01,
                 replay_memory_size=10000, batch_size=32, target_update_freq=100,
                 alpha=0.6, beta=0.4, beta_increment=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = min_exploration
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.beta = beta
        self.beta_increment = beta_increment
        
        # Prioritized replay buffer
        self.memory = PrioritizedReplayBuffer(replay_memory_size, alpha)
        
        # Main network and target network
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()
        
        self.training_step = 0
        
    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr))
        return model
    
    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        # Initial priority is maximum priority
        max_priority = self.memory.priorities.max() if self.memory.size > 0 else 1.0
        self.memory.add((state, action, reward, next_state, done), max_priority)
    
    def choose_action(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = np.reshape(state, [1, self.state_size])
            q_values = self.model.predict(state, verbose=0)
            return np.argmax(q_values[0])
    
    def replay(self):
        if self.memory.size < self.batch_size:
            return
        
        # Sample from prioritized replay buffer
        experiences, indices, weights = self.memory.sample(self.batch_size, self.beta)
        
        states = np.array([exp[0] for exp in experiences])
        actions = np.array([exp[1] for exp in experiences])
        rewards = np.array([exp[2] for exp in experiences])
        next_states = np.array([exp[3] for exp in experiences])
        dones = np.array([exp[4] for exp in experiences])
        
        # Current Q values
        current_q = self.model.predict(states, verbose=0)
        
        # Next Q values from target network
        next_q = self.target_model.predict(next_states, verbose=0)
        
        # Update Q values and calculate TD errors
        target_q = current_q.copy()
        td_errors = np.zeros(self.batch_size)
        
        for i in range(self.batch_size):
            if dones[i]:
                target = rewards[i]
            else:
                target = rewards[i] + self.gamma * np.max(next_q[i])
            
            td_error = abs(target - current_q[i, actions[i]])
            td_errors[i] = td_error
            target_q[i, actions[i]] = target
        
        # Update priorities
        self.memory.update_priorities(indices, td_errors + 1e-6)
        
        # Train the model with importance sampling weights
        self.model.fit(states, target_q, sample_weight=weights, epochs=1, verbose=0)
        
        # Update exploration rate and beta
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()
    
    def train(self, env, episodes=1000, max_steps=500):
        scores = []
        
        for episode in range(episodes):
            state = env.reset()
            state = np.reshape(state, [1, self.state_size])
            total_reward = 0
            
            for step in range(max_steps):
                action = self.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                if done:
                    scores.append(total_reward)
                    break
            
            self.replay()
            
            if episode % 100 == 0:
                avg_score = np.mean(scores[-100:]) if scores else 0
                print(f"Episode {episode}, Avg Score: {avg_score:.2f}, Epsilon: {self.epsilon:.3f}")
    
    def save_model(self, filepath):
        self.model.save(filepath)
    
    def load_model(self, filepath):
        self.model = tf.keras.models.load_model(filepath)
        self.update_target_network()