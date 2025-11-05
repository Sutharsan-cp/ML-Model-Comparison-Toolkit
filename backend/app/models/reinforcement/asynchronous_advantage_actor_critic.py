import numpy as np
import tensorflow as tf
import threading
import multiprocessing
from collections import deque

class A3C:
    def __init__(self, state_size, action_size, actor_lr=0.0001, critic_lr=0.0005,
                 discount_factor=0.99, entropy_coef=0.01, max_episode_steps=500,
                 n_workers=multiprocessing.cpu_count()):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = discount_factor
        self.entropy_coef = entropy_coef
        self.max_episode_steps = max_episode_steps
        self.n_workers = n_workers
        
        # Global network
        self.global_actor = self._build_actor()
        self.global_critic = self._build_critic()
        
        # Global optimizer
        self.global_actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.global_critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        
        self.lock = threading.Lock()
        self.global_episode = 0
        self.results = deque(maxlen=100)
    
    def _build_actor(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='softmax')
        ])
        return model
    
    def _build_critic(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        return model
    
    def train(self, env, episodes=1000):
        workers = []
        
        for worker_id in range(self.n_workers):
            worker = A3CWorker(worker_id, self, env)
            workers.append(worker)
            worker.start()
        
        for worker in workers:
            worker.join()
    
    def update_global_network(self, actor_grads, critic_grads):
        with self.lock:
            self.global_actor_optimizer.apply_gradients(
                zip(actor_grads, self.global_actor.trainable_variables)
            )
            self.global_critic_optimizer.apply_gradients(
                zip(critic_grads, self.global_critic.trainable_variables)
            )

class A3CWorker(threading.Thread):
    def __init__(self, worker_id, global_network, env):
        super(A3CWorker, self).__init__()
        self.worker_id = worker_id
        self.global_network = global_network
        self.env = env
        
        # Local network
        self.local_actor = self._build_actor()
        self.local_critic = self._build_critic()
        
        self.sync_with_global()
    
    def _build_actor(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.global_network.state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.global_network.action_size, activation='softmax')
        ])
        return model
    
    def _build_critic(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.global_network.state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        return model
    
    def sync_with_global(self):
        self.local_actor.set_weights(self.global_network.global_actor.get_weights())
        self.local_critic.set_weights(self.global_network.global_critic.get_weights())
    
    def choose_action(self, state):
        state = np.reshape(state, [1, self.global_network.state_size])
        probabilities = self.local_actor.predict(state, verbose=0)[0]
        action = np.random.choice(self.global_network.action_size, p=probabilities)
        return action
    
    def compute_advantages(self, rewards, values, next_value, dones):
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        R = next_value
        for t in reversed(range(len(rewards))):
            R = rewards[t] + self.global_network.gamma * R * (1 - dones[t])
            returns[t] = R
            advantages[t] = R - values[t]
        
        return advantages, returns
    
    def run(self):
        while self.global_network.global_episode < 1000:  # Training episodes
            state = self.env.reset()
            done = False
            episode_reward = 0
            step = 0
            
            states, actions, rewards, values, dones = [], [], [], [], []
            
            while not done and step < self.global_network.max_episode_steps:
                # Sync local network with global
                self.sync_with_global()
                
                # Choose action
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                # Store transition
                state_tensor = np.reshape(state, [1, self.global_network.state_size])
                value = self.local_critic.predict(state_tensor, verbose=0)[0, 0]
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                values.append(value)
                dones.append(done)
                
                state = next_state
                episode_reward += reward
                step += 1
            
            # Compute advantages and returns
            next_state_tensor = np.reshape(state, [1, self.global_network.state_size])
            next_value = self.local_critic.predict(next_state_tensor, verbose=0)[0, 0] if not done else 0
            
            advantages, returns = self.compute_advantages(rewards, values, next_value, dones)
            
            # Normalize advantages
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
            
            # Convert to tensors
            states = np.array(states)
            actions = np.array(actions)
            returns = np.array(returns).reshape(-1, 1)
            advantages = np.array(advantages)
            
            # Compute gradients
            with tf.GradientTape(persistent=True) as tape:
                # Actor loss
                action_probs = self.local_actor(states, training=True)
                action_mask = tf.one_hot(actions, self.global_network.action_size)
                chosen_action_probs = tf.reduce_sum(action_probs * action_mask, axis=1)
                
                actor_loss = -tf.reduce_mean(tf.math.log(chosen_action_probs + 1e-8) * advantages)
                
                # Entropy bonus
                entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-8), axis=1)
                entropy_loss = -tf.reduce_mean(entropy)
                
                total_actor_loss = actor_loss + self.global_network.entropy_coef * entropy_loss
                
                # Critic loss
                current_values = self.local_critic(states, training=True)
                critic_loss = tf.keras.losses.MSE(returns, current_values)
            
            # Compute gradients
            actor_grads = tape.gradient(total_actor_loss, self.local_actor.trainable_variables)
            critic_grads = tape.gradient(critic_loss, self.local_critic.trainable_variables)
            
            del tape
            
            # Update global network
            self.global_network.update_global_network(actor_grads, critic_grads)
            
            # Update global episode counter
            with self.global_network.lock:
                self.global_network.global_episode += 1
                self.global_network.results.append(episode_reward)
                
                if self.global_network.global_episode % 100 == 0:
                    avg_reward = np.mean(self.global_network.results)
                    print(f"Global Episode: {self.global_network.global_episode}, "
                          f"Avg Reward: {avg_reward:.2f}, Worker: {self.worker_id}")