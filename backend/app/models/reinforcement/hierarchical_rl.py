import numpy as np
import tensorflow as tf
from collections import defaultdict

class HierarchicalRL:
    def __init__(self, state_size, subgoal_size, low_level_action_size,
                 high_level_lr=0.001, low_level_lr=0.001, discount_factor=0.99,
                 intrinsic_reward_coef=1.0, subgoal_test_period=10):
        self.state_size = state_size
        self.subgoal_size = subgoal_size
        self.low_level_action_size = low_level_action_size
        self.gamma = discount_factor
        self.intrinsic_reward_coef = intrinsic_reward_coef
        self.subgoal_test_period = subgoal_test_period
        
        # High-level controller (manager)
        self.high_level_policy = self._build_high_level_policy()
        self.high_level_optimizer = tf.keras.optimizers.Adam(learning_rate=high_level_lr)
        
        # Low-level controller (worker)
        self.low_level_policy = self._build_low_level_policy()
        self.low_level_optimizer = tf.keras.optimizers.Adam(learning_rate=low_level_lr)
        
        # Experience buffers
        self.high_level_buffer = []
        self.low_level_buffer = []
        
        self.training_history = []
    
    def _build_high_level_policy(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.subgoal_size, activation='tanh')
        ])
        return model
    
    def _build_low_level_policy(self):
        # Input: state + subgoal
        inputs = tf.keras.layers.Input(shape=(self.state_size + self.subgoal_size,))
        x = tf.keras.layers.Dense(64, activation='relu')(inputs)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(self.low_level_action_size, activation='softmax')(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def high_level_choose_subgoal(self, state):
        state = np.reshape(state, [1, self.state_size])
        subgoal = self.high_level_policy.predict(state, verbose=0)[0]
        return subgoal
    
    def low_level_choose_action(self, state, subgoal):
        combined_input = np.concatenate([state, subgoal])
        combined_input = np.reshape(combined_input, [1, self.state_size + self.subgoal_size])
        action_probs = self.low_level_policy.predict(combined_input, verbose=0)[0]
        action = np.random.choice(self.low_level_action_size, p=action_probs)
        return action
    
    def compute_intrinsic_reward(self, state, subgoal):
        """Compute intrinsic reward based on subgoal achievement"""
        # Simple Euclidean distance-based intrinsic reward
        state_achievement = state[:self.subgoal_size]  # Assume first subgoal_size dimensions are relevant
        distance = np.linalg.norm(state_achievement - subgoal)
        intrinsic_reward = -distance  # Negative distance as reward (closer is better)
        return intrinsic_reward * self.intrinsic_reward_coef
    
    def store_high_level_transition(self, state, subgoal, intrinsic_reward, next_state, done):
        self.high_level_buffer.append((state, subgoal, intrinsic_reward, next_state, done))
    
    def store_low_level_transition(self, state, subgoal, action, extrinsic_reward, next_state, done):
        self.low_level_buffer.append((state, subgoal, action, extrinsic_reward, next_state, done))
    
    def train_high_level(self):
        if len(self.high_level_buffer) == 0:
            return
        
        states, subgoals, intrinsic_rewards, next_states, dones = zip(*self.high_level_buffer)
        
        states = np.array(states)
        intrinsic_rewards = np.array(intrinsic_rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        # Compute returns
        returns = np.zeros_like(intrinsic_rewards)
        G = 0
        for t in reversed(range(len(intrinsic_rewards))):
            G = intrinsic_rewards[t] + self.gamma * G * (1 - dones[t])
            returns[t] = G
        
        # Normalize returns
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        
        with tf.GradientTape() as tape:
            # Policy gradient update for high-level policy
            predicted_subgoals = self.high_level_policy(states, training=True)
            
            # Simple MSE loss between predicted and actual subgoals weighted by returns
            # In practice, you might want a more sophisticated approach
            high_level_loss = -tf.reduce_mean(returns * tf.reduce_sum(predicted_subgoals * subgoals, axis=1))
        
        grads = tape.gradient(high_level_loss, self.high_level_policy.trainable_variables)
        self.high_level_optimizer.apply_gradients(zip(grads, self.high_level_policy.trainable_variables))
        
        # Clear buffer
        self.high_level_buffer = []
    
    def train_low_level(self):
        if len(self.low_level_buffer) == 0:
            return
        
        states, subgoals, actions, extrinsic_rewards, next_states, dones = zip(*self.low_level_buffer)
        
        states = np.array(states)
        subgoals = np.array(subgoals)
        actions = np.array(actions)
        extrinsic_rewards = np.array(extrinsic_rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        # Combine states and subgoals
        combined_inputs = np.concatenate([states, subgoals], axis=1)
        
        # Compute returns
        returns = np.zeros_like(extrinsic_rewards)
        G = 0
        for t in reversed(range(len(extrinsic_rewards))):
            G = extrinsic_rewards[t] + self.gamma * G * (1 - dones[t])
            returns[t] = G
        
        # Normalize returns
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        
        with tf.GradientTape() as tape:
            # Policy gradient update for low-level policy
            action_probs = self.low_level_policy(combined_inputs, training=True)
            action_mask = tf.one_hot(actions, self.low_level_action_size)
            chosen_action_probs = tf.reduce_sum(action_probs * action_mask, axis=1)
            
            low_level_loss = -tf.reduce_mean(tf.math.log(chosen_action_probs + 1e-8) * returns)
        
        grads = tape.gradient(low_level_loss, self.low_level_policy.trainable_variables)
        self.low_level_optimizer.apply_gradients(zip(grads, self.low_level_policy.trainable_variables))
        
        # Clear buffer
        self.low_level_buffer = []
    
    def train(self, env, episodes=1000, max_steps=500, subgoal_horizon=10):
        scores = []
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            step = 0
            
            while step < max_steps:
                # High-level: choose subgoal
                subgoal = self.high_level_choose_subgoal(state)
                subgoal_start_state = state.copy()
                
                # Low-level: execute actions to achieve subgoal
                for _ in range(subgoal_horizon):
                    if step >= max_steps:
                        break
                    
                    action = self.low_level_choose_action(state, subgoal)
                    next_state, extrinsic_reward, done, _ = env.step(action)
                    
                    # Compute intrinsic reward
                    intrinsic_reward = self.compute_intrinsic_reward(next_state, subgoal)
                    
                    # Store low-level transition
                    self.store_low_level_transition(state, subgoal, action, extrinsic_reward, next_state, done)
                    
                    state = next_state
                    total_reward += extrinsic_reward
                    step += 1
                    
                    if done:
                        break
                
                # Store high-level transition
                self.store_high_level_transition(subgoal_start_state, subgoal, intrinsic_reward, state, done)
                
                # Train both levels
                self.train_low_level()
                self.train_high_level()
                
                if done:
                    break
            
            scores.append(total_reward)
            
            if episode % 100 == 0:
                avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
                print(f"Episode {episode}, Avg Score: {avg_score:.2f}")
    
    def save_models(self, high_level_path, low_level_path):
        self.high_level_policy.save(high_level_path)
        self.low_level_policy.save(low_level_path)
    
    def load_models(self, high_level_path, low_level_path):
        self.high_level_policy = tf.keras.models.load_model(high_level_path)
        self.low_level_policy = tf.keras.models.load_model(low_level_path)