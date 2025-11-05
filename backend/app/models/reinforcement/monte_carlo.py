import numpy as np
import random
from collections import defaultdict

class MonteCarlo:
    def __init__(self, n_states, n_actions, discount_factor=0.99, exploration_rate=1.0,
                 exploration_decay=0.995, min_exploration=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = min_exploration
        
        # Initialize Q-table and returns
        self.q_table = np.zeros((n_states, n_actions))
        self.returns = defaultdict(list)
        self.visit_count = np.zeros((n_states, n_actions))
        
        self.training_history = []
    
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return np.argmax(self.q_table[state])
    
    def generate_episode(self, env, max_steps=100):
        episode = []
        state = env.reset()
        
        for step in range(max_steps):
            action = self.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            
            episode.append((state, action, reward))
            
            if done:
                break
            state = next_state
        
        return episode
    
    def update_q_values(self, episode):
        # Calculate returns for each state-action pair
        g = 0
        visited_state_actions = set()
        
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            g = self.gamma * g + reward
            
            # First visit MC: only update the first time we visit a state-action pair
            if (state, action) not in visited_state_actions:
                visited_state_actions.add((state, action))
                self.returns[(state, action)].append(g)
                
                # Update Q-value as average of returns
                self.q_table[state, action] = np.mean(self.returns[(state, action)])
                self.visit_count[state, action] += 1
    
    def train(self, env, episodes=1000, max_steps=100):
        for episode_num in range(episodes):
            episode = self.generate_episode(env, max_steps)
            total_reward = sum([step[2] for step in episode])
            
            self.update_q_values(episode)
            
            # Decay exploration rate
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            self.training_history.append({
                'episode': episode_num,
                'total_reward': total_reward,
                'epsilon': self.epsilon,
                'steps': len(episode)
            })
            
            if episode_num % 100 == 0:
                avg_reward = np.mean([h['total_reward'] for h in self.training_history[-100:]])
                print(f"Episode {episode_num}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")
    
    def get_policy(self):
        return np.argmax(self.q_table, axis=1)
    
    def get_value_function(self):
        return np.max(self.q_table, axis=1)
    
    def get_visit_counts(self):
        return self.visit_count
    
    def save_model(self, filepath):
        np.save(filepath, self.q_table)
    
    def load_model(self, filepath):
        self.q_table = np.load(filepath)