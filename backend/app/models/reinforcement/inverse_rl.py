import numpy as np
import tensorflow as tf
from collections import defaultdict

class InverseRL:
    def __init__(self, state_size, action_size, learning_rate=0.001, 
                 discount_factor=0.99, n_trajectories=100):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.n_trajectories = n_trajectories
        
        # Reward function network
        self.reward_network = self._build_reward_network()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        
        # Expert trajectories
        self.expert_trajectories = []
        
        # Policy for sampling trajectories (can be any RL algorithm)
        self.policy_network = self._build_policy_network()
    
    def _build_reward_network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        return model
    
    def _build_policy_network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='softmax')
        ])
        return model
    
    def add_expert_trajectory(self, trajectory):
        """Add an expert trajectory (list of state-action pairs)"""
        self.expert_trajectories.append(trajectory)
    
    def compute_feature_expectations(self, trajectories):
        """Compute feature expectations from trajectories"""
        if len(trajectories) == 0:
            return np.zeros(self.state_size)
        
        feature_expectations = np.zeros(self.state_size)
        total_discount = 0
        
        for trajectory in trajectories:
            discount = 1.0
            for state, action, _ in trajectory:
                feature_expectations += state * discount
                total_discount += discount
                discount *= self.gamma
        
        if total_discount > 0:
            feature_expectations /= total_discount
        
        return feature_expectations
    
    def sample_trajectories(self, env, policy, n_trajectories, max_steps=100):
        """Sample trajectories using the current policy"""
        trajectories = []
        
        for _ in range(n_trajectories):
            state = env.reset()
            trajectory = []
            
            for step in range(max_steps):
                # Choose action using policy
                state_tensor = np.reshape(state, [1, self.state_size])
                action_probs = policy.predict(state_tensor, verbose=0)[0]
                action = np.random.choice(self.action_size, p=action_probs)
                
                # Take action
                next_state, reward, done, _ = env.step(action)
                
                trajectory.append((state, action, reward))
                state = next_state
                
                if done:
                    break
            
            trajectories.append(trajectory)
        
        return trajectories
    
    def compute_reward(self, state):
        """Compute reward for a given state"""
        state = np.reshape(state, [1, self.state_size])
        return self.reward_network.predict(state, verbose=0)[0, 0]
    
    def train(self, env, iterations=1000, n_policy_trajectories=50, max_steps=100):
        """Maximum Entropy Inverse RL training"""
        
        # Compute expert feature expectations
        expert_features = self.compute_feature_expectations(self.expert_trajectories)
        
        # Initialize policy
        policy_trajectories = self.sample_trajectories(env, self.policy_network, 
                                                      n_policy_trajectories, max_steps)
        policy_features = self.compute_feature_expectations(policy_trajectories)
        
        for iteration in range(iterations):
            # Update reward function weights to maximize likelihood
            with tf.GradientTape() as tape:
                # Compute expert trajectory rewards
                expert_rewards = 0
                expert_discount = 0
                for trajectory in self.expert_trajectories:
                    discount = 1.0
                    for state, action, _ in trajectory:
                        reward = self.reward_network(np.reshape(state, [1, self.state_size]), training=True)
                        expert_rewards += reward * discount
                        expert_discount += discount
                        discount *= self.gamma
                
                if expert_discount > 0:
                    expert_rewards /= expert_discount
                
                # Compute policy trajectory rewards
                policy_rewards = 0
                policy_discount = 0
                for trajectory in policy_trajectories:
                    discount = 1.0
                    for state, action, _ in trajectory:
                        reward = self.reward_network(np.reshape(state, [1, self.state_size]), training=True)
                        policy_rewards += reward * discount
                        policy_discount += discount
                        discount *= self.gamma
                
                if policy_discount > 0:
                    policy_rewards /= policy_discount
                
                # Maximum entropy loss
                loss = policy_rewards - expert_rewards
            
            # Update reward network
            grads = tape.gradient(loss, self.reward_network.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.reward_network.trainable_variables))
            
            # Improve policy using the new reward function
            self.improve_policy(env, n_policy_trajectories, max_steps)
            
            # Sample new trajectories with updated policy
            policy_trajectories = self.sample_trajectories(env, self.policy_network, 
                                                          n_policy_trajectories, max_steps)
            policy_features = self.compute_feature_expectations(policy_trajectories)
            
            # Check convergence
            feature_diff = np.linalg.norm(expert_features - policy_features)
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Feature Difference: {feature_diff:.4f}")
            
            if feature_diff < 0.01:  # Convergence threshold
                print("Converged!")
                break
    
    def improve_policy(self, env, n_trajectories, max_steps):
        """Improve policy using policy gradient with the learned reward function"""
        # This is a simplified policy improvement step
        # In practice, you might use a more sophisticated RL algorithm
        
        states, actions, rewards = [], [], []
        
        # Collect data with current policy
        for _ in range(n_trajectories):
            state = env.reset()
            
            for step in range(max_steps):
                # Choose action
                state_tensor = np.reshape(state, [1, self.state_size])
                action_probs = self.policy_network.predict(state_tensor, verbose=0)[0]
                action = np.random.choice(self.action_size, p=action_probs)
                
                # Get reward from learned reward function
                reward = self.compute_reward(state)
                
                # Take action
                next_state, _, done, _ = env.step(action)
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                
                state = next_state
                
                if done:
                    break
        
        # Policy gradient update
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        
        # Normalize rewards
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
        
        with tf.GradientTape() as tape:
            action_probs = self.policy_network(states, training=True)
            action_mask = tf.one_hot(actions, self.action_size)
            chosen_action_probs = tf.reduce_sum(action_probs * action_mask, axis=1)
            
            policy_loss = -tf.reduce_mean(tf.math.log(chosen_action_probs + 1e-8) * rewards)
        
        grads = tape.gradient(policy_loss, self.policy_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy_network.trainable_variables))
    
    def get_reward_function(self):
        return self.reward_network
    
    def save_models(self, reward_path, policy_path):
        self.reward_network.save(reward_path)
        self.policy_network.save(policy_path)
    
    def load_models(self, reward_path, policy_path):
        self.reward_network = tf.keras.models.load_model(reward_path)
        self.policy_network = tf.keras.models.load_model(policy_path)