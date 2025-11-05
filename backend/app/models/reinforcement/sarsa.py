import numpy as np
import random
from collections import defaultdict

class SARSA:
    def __init__(self, n_states, n_actions, learning_rate=0.1, discount_factor=0.99,
                 exploration_rate=1.0, exploration_decay=0.995, min_exploration=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = min_exploration
        
        # Initialize Q-table
        self.q_table = np.zeros((n_states, n_actions))
        
        # Training history
        self.training_history = []
    
    def choose_action(self, state):
        if random.random() < self.epsilon:
            # Exploration: random action
            return random.randint(0, self.n_actions - 1)
        else:
            # Exploitation: best action from Q-table
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, next_action, done):
        # Current Q-value
        current_q = self.q_table[state, action]
        
        if done:
            target = reward
        else:
            # SARSA: use next action (on-policy)
            next_q = self.q_table[next_state, next_action]
            target = reward + self.gamma * next_q
        
        # Update Q-value
        self.q_table[state, action] = current_q + self.lr * (target - current_q)
        
        # Decay exploration rate
        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def train(self, env, episodes=1000, max_steps=100):
        for episode in range(episodes):
            state = env.reset()
            action = self.choose_action(state)
            total_reward = 0
            
            for step in range(max_steps):
                next_state, reward, done, _ = env.step(action)
                next_action = self.choose_action(next_state)
                
                self.update(state, action, reward, next_state, next_action, done)
                
                state = next_state
                action = next_action
                total_reward += reward
                
                if done:
                    break
            
            self.training_history.append({
                'episode': episode,
                'total_reward': total_reward,
                'epsilon': self.epsilon,
                'steps': step + 1
            })
            
            if episode % 100 == 0:
                avg_reward = np.mean([h['total_reward'] for h in self.training_history[-100:]])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")
    
    def get_policy(self):
        return np.argmax(self.q_table, axis=1)
    
    def get_value_function(self):
        return np.max(self.q_table, axis=1)
    
    def save_model(self, filepath):
        np.save(filepath, self.q_table)
    
    def load_model(self, filepath):
        self.q_table = np.load(filepath)