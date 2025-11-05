import numpy as np
import random
import tensorflow as tf
from collections import deque

class DuelingDQN:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.99,
                 exploration_rate=1.0, exploration_decay=0.995, min_exploration=0.01,
                 replay_memory_size=10000, batch_size=32, target_update_freq=100):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = min_exploration
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Replay memory
        self.memory = deque(maxlen=replay_memory_size)
        
        # Main network and target network with dueling architecture
        self.model = self._build_dueling_model()
        self.target_model = self._build_dueling_model()
        self.update_target_network()
        
        self.training_step = 0
        
    def _build_dueling_model(self):
        inputs = tf.keras.layers.Input(shape=(self.state_size,))
        
        # Common layers
        x = tf.keras.layers.Dense(24, activation='relu')(inputs)
        x = tf.keras.layers.Dense(24, activation='relu')(x)
        
        # Value stream
        value_stream = tf.keras.layers.Dense(24, activation='relu')(x)
        value = tf.keras.layers.Dense(1, activation='linear')(value_stream)
        
        # Advantage stream
        advantage_stream = tf.keras.layers.Dense(24, activation='relu')(x)
        advantage = tf.keras.layers.Dense(self.action_size, activation='linear')(advantage_stream)
        
        # Combine value and advantage
        q_values = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
        
        model = tf.keras.Model(inputs=inputs, outputs=q_values)
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr))
        return model
    
    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def choose_action(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = np.reshape(state, [1, self.state_size])
            q_values = self.model.predict(state, verbose=0)
            return np.argmax(q_values[0])
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([experience[0] for experience in batch])
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        next_states = np.array([experience[3] for experience in batch])
        dones = np.array([experience[4] for experience in batch])
        
        # Current Q values
        current_q = self.model.predict(states, verbose=0)
        
        # Next Q values from target network
        next_q = self.target_model.predict(next_states, verbose=0)
        
        # Update Q values
        target_q = current_q.copy()
        for i in range(self.batch_size):
            if dones[i]:
                target_q[i, actions[i]] = rewards[i]
            else:
                target_q[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])
        
        # Train the model
        self.model.fit(states, target_q, epochs=1, verbose=0)
        
        # Update exploration rate
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
        
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