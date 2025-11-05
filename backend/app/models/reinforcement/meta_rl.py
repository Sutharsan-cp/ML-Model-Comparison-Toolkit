import numpy as np
import tensorflow as tf
from collections import defaultdict

class MAML:
    def __init__(self, state_size, action_size, inner_lr=0.1, outer_lr=0.001,
                 discount_factor=0.99, adaptation_steps=1, num_tasks=4):
        self.state_size = state_size
        self.action_size = action_size
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.gamma = discount_factor
        self.adaptation_steps = adaptation_steps
        self.num_tasks = num_tasks
        
        # Meta-policy network
        self.meta_policy = self._build_policy_network()
        self.outer_optimizer = tf.keras.optimizers.Adam(learning_rate=outer_lr)
        
        self.training_history = []
    
    def _build_policy_network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='softmax')
        ])
        return model
    
    def sample_task(self, env_pool):
        """Sample a task from the environment pool"""
        return np.random.choice(env_pool, 1)[0]
    
    def adapt(self, env, support_set, adapted_policy):
        """Adapt the policy using the support set"""
        if len(support_set) == 0:
            return adapted_policy
        
        states, actions, rewards = zip(*support_set)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        
        # Normalize rewards
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
        
        # Compute gradients for adaptation
        with tf.GradientTape() as tape:
            action_probs = adapted_policy(states, training=True)
            action_mask = tf.one_hot(actions, self.action_size)
            chosen_action_probs = tf.reduce_sum(action_probs * action_mask, axis=1)
            
            loss = -tf.reduce_mean(tf.math.log(chosen_action_probs + 1e-8) * rewards)
        
        grads = tape.gradient(loss, adapted_policy.trainable_variables)
        
        # Apply gradient descent step
        updated_weights = []
        for weights, grad in zip(adapted_policy.get_weights(), grads):
            if grad is not None:
                updated_weights.append(weights - self.inner_lr * grad)
            else:
                updated_weights.append(weights)
        
        adapted_policy.set_weights(updated_weights)
        return adapted_policy
    
    def compute_loss(self, env, query_set, adapted_policy):
        """Compute loss on query set using adapted policy"""
        if len(query_set) == 0:
            return 0.0
        
        states, actions, rewards = zip(*query_set)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        
        # Normalize rewards
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
        
        with tf.GradientTape() as tape:
            action_probs = adapted_policy(states, training=True)
            action_mask = tf.one_hot(actions, self.action_size)
            chosen_action_probs = tf.reduce_sum(action_probs * action_mask, axis=1)
            
            loss = -tf.reduce_mean(tf.math.log(chosen_action_probs + 1e-8) * rewards)
        
        return loss
    
    def sample_trajectory(self, env, policy, max_steps=100):
        """Sample a trajectory using the given policy"""
        state = env.reset()
        trajectory = []
        
        for step in range(max_steps):
            # Choose action
            state_tensor = np.reshape(state, [1, self.state_size])
            action_probs = policy.predict(state_tensor, verbose=0)[0]
            action = np.random.choice(self.action_size, p=action_probs)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            trajectory.append((state, action, reward))
            state = next_state
            
            if done:
                break
        
        return trajectory
    
    def train(self, env_pool, meta_iterations=1000, shots=10, query_size=10, max_steps=100):
        """Model-Agnostic Meta-Learning training"""
        
        for iteration in range(meta_iterations):
            total_meta_loss = 0
            
            with tf.GradientTape() as meta_tape:
                # Sample batch of tasks
                tasks = [self.sample_task(env_pool) for _ in range(self.num_tasks)]
                
                task_losses = []
                
                for task in tasks:
                    # Create adapted policy for this task
                    adapted_policy = self._build_policy_network()
                    adapted_policy.set_weights(self.meta_policy.get_weights())
                    
                    # Sample support and query sets
                    support_set = []
                    for _ in range(shots):
                        trajectory = self.sample_trajectory(task, adapted_policy, max_steps)
                        support_set.extend(trajectory)
                    
                    # Adaptation steps
                    for _ in range(self.adaptation_steps):
                        adapted_policy = self.adapt(task, support_set, adapted_policy)
                    
                    # Sample query set
                    query_set = []
                    for _ in range(query_size):
                        trajectory = self.sample_trajectory(task, adapted_policy, max_steps)
                        query_set.extend(trajectory)
                    
                    # Compute loss on query set
                    query_loss = self.compute_loss(task, query_set, adapted_policy)
                    task_losses.append(query_loss)
                
                # Meta-loss is average over tasks
                meta_loss = tf.reduce_mean(task_losses)
                total_meta_loss += meta_loss.numpy()
            
            # Meta-optimization step
            meta_grads = meta_tape.gradient(meta_loss, self.meta_policy.trainable_variables)
            self.outer_optimizer.apply_gradients(zip(meta_grads, self.meta_policy.trainable_variables))
            
            self.training_history.append({
                'iteration': iteration,
                'meta_loss': total_meta_loss / self.num_tasks
            })
            
            if iteration % 100 == 0:
                print(f"Meta Iteration {iteration}, Meta Loss: {total_meta_loss / self.num_tasks:.4f}")
    
    def fast_adapt(self, env, support_trajectories, adaptation_steps=None):
        """Quick adaptation to a new task"""
        if adaptation_steps is None:
            adaptation_steps = self.adaptation_steps
        
        adapted_policy = self._build_policy_network()
        adapted_policy.set_weights(self.meta_policy.get_weights())
        
        support_set = []
        for trajectory in support_trajectories:
            support_set.extend(trajectory)
        
        for _ in range(adaptation_steps):
            adapted_policy = self.adapt(env, support_set, adapted_policy)
        
        return adapted_policy
    
    def evaluate(self, env_pool, test_tasks=10, shots=5, adaptation_steps=None):
        """Evaluate meta-policy on test tasks"""
        test_scores = []
        
        for _ in range(test_tasks):
            task = self.sample_task(env_pool)
            
            # Sample support trajectories
            support_trajectories = []
            for _ in range(shots):
                trajectory = self.sample_trajectory(task, self.meta_policy)
                support_trajectories.append(trajectory)
            
            # Fast adaptation
            adapted_policy = self.fast_adapt(task, support_trajectories, adaptation_steps)
            
            # Evaluate adapted policy
            total_reward = 0
            for _ in range(5):  # Multiple test episodes
                trajectory = self.sample_trajectory(task, adapted_policy)
                episode_reward = sum([step[2] for step in trajectory])
                total_reward += episode_reward
            
            test_scores.append(total_reward / 5)
        
        return np.mean(test_scores)
    
    def save_meta_policy(self, filepath):
        self.meta_policy.save(filepath)
    
    def load_meta_policy(self, filepath):
        self.meta_policy = tf.keras.models.load_model(filepath)