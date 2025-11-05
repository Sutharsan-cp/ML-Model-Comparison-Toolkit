import numpy as np
import random
import tensorflow as tf
from collections import deque

class SAC:
    def __init__(self, state_size, action_size, action_low, action_high,
                 actor_lr=0.0003, critic_lr=0.0003, alpha_lr=0.0003,
                 discount_factor=0.99, replay_memory_size=10000, batch_size=256,
                 tau=0.005, alpha=0.2, auto_entropy_tuning=True, target_entropy=None):
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.lr_actor = actor_lr
        self.lr_critic = critic_lr
        self.lr_alpha = alpha_lr
        self.gamma = discount_factor
        self.batch_size = batch_size
        self.tau = tau
        self.auto_entropy_tuning = auto_entropy_tuning
        
        # Replay memory
        self.memory = deque(maxlen=replay_memory_size)
        
        # Actor and Critic networks
        self.actor = self._build_actor()
        self.critic1 = self._build_critic()
        self.critic2 = self._build_critic()
        
        # Target networks
        self.target_critic1 = self._build_critic()
        self.target_critic2 = self._build_critic()
        
        # Initialize target networks
        self.update_target_networks(tau=1.0)
        
        # Temperature parameter
        self.alpha = tf.Variable(alpha, dtype=tf.float32)
        if self.auto_entropy_tuning:
            self.target_entropy = target_entropy or -np.prod(action_size)
            self.log_alpha = tf.Variable(tf.math.log(alpha), dtype=tf.float32)
            self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=alpha_lr)
        
        # Optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic1_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        self.critic2_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
    
    def _build_actor(self):
        inputs = tf.keras.layers.Input(shape=(self.state_size,))
        x = tf.keras.layers.Dense(256, activation='relu')(inputs)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        
        # Mean and log std
        mean = tf.keras.layers.Dense(self.action_size, activation='tanh')(x)
        log_std = tf.keras.layers.Dense(self.action_size)(x)
        log_std = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, -20, 2))(log_std)
        
        # Scale mean to action space
        mean = tf.keras.layers.Lambda(
            lambda x: x * (self.action_high - self.action_low) / 2 + 
                     (self.action_high + self.action_low) / 2
        )(mean)
        
        return tf.keras.Model(inputs=inputs, outputs=[mean, log_std])
    
    def _build_critic(self):
        # State input
        state_input = tf.keras.layers.Input(shape=(self.state_size,))
        state_out = tf.keras.layers.Dense(256, activation='relu')(state_input)
        state_out = tf.keras.layers.Dense(256, activation='relu')(state_out)
        
        # Action input
        action_input = tf.keras.layers.Input(shape=(self.action_size,))
        action_out = tf.keras.layers.Dense(256, activation='relu')(action_input)
        
        # Combine state and action
        concat = tf.keras.layers.Concatenate()([state_out, action_out])
        x = tf.keras.layers.Dense(256, activation='relu')(concat)
        outputs = tf.keras.layers.Dense(1, activation='linear')(x)
        
        return tf.keras.Model(inputs=[state_input, action_input], outputs=outputs)
    
    def sample_action(self, state):
        state = np.reshape(state, [1, self.state_size])
        mean, log_std = self.actor(state, training=False)
        std = tf.exp(log_std)
        
        # Reparameterization trick
        normal = tf.random.normal(shape=mean.shape)
        action = mean + std * normal
        
        # Apply tanh squashing
        action = tf.tanh(action)
        
        # Scale to action space
        action = action * (self.action_high - self.action_low) / 2 + \
                (self.action_high + self.action_low) / 2
        
        return action.numpy()[0]
    
    def update_target_networks(self, tau=None):
        tau = tau or self.tau
        
        # Update target critics
        critic1_weights = self.critic1.get_weights()
        target_critic1_weights = self.target_critic1.get_weights()
        critic2_weights = self.critic2.get_weights()
        target_critic2_weights = self.target_critic2.get_weights()
        
        for i in range(len(critic1_weights)):
            target_critic1_weights[i] = tau * critic1_weights[i] + (1 - tau) * target_critic1_weights[i]
            target_critic2_weights[i] = tau * critic2_weights[i] + (1 - tau) * target_critic2_weights[i]
        
        self.target_critic1.set_weights(target_critic1_weights)
        self.target_critic2.set_weights(target_critic2_weights)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([experience[0] for experience in batch])
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        next_states = np.array([experience[3] for experience in batch])
        dones = np.array([experience[4] for experience in batch])
        
        # Update critics
        with tf.GradientTape(persistent=True) as tape:
            # Sample next actions and compute target Q values
            next_mean, next_log_std = self.actor(next_states, training=True)
            next_std = tf.exp(next_log_std)
            next_normal = tf.random.normal(shape=next_mean.shape)
            next_actions = next_mean + next_std * next_normal
            next_actions = tf.tanh(next_actions)
            
            # Scale next actions
            next_actions = next_actions * (self.action_high - self.action_low) / 2 + \
                          (self.action_high + self.action_low) / 2
            
            # Target Q values
            target_q1 = self.target_critic1([next_states, next_actions], training=True)
            target_q2 = self.target_critic2([next_states, next_actions], training=True)
            target_q = tf.minimum(target_q1, target_q2)
            
            # Compute log probability
            log_prob = -0.5 * tf.reduce_sum(tf.square((next_actions - next_mean) / next_std) + 
                                          2 * next_log_std + np.log(2 * np.pi), axis=1, keepdims=True)
            
            target_q = rewards + self.gamma * (1 - dones) * (target_q - self.alpha * log_prob)
            
            # Current Q values
            current_q1 = self.critic1([states, actions], training=True)
            current_q2 = self.critic2([states, actions], training=True)
            
            critic1_loss = tf.keras.losses.MSE(target_q, current_q1)
            critic2_loss = tf.keras.losses.MSE(target_q, current_q2)
        
        # Update critics
        critic1_grads = tape.gradient(critic1_loss, self.critic1.trainable_variables)
        self.critic1_optimizer.apply_gradients(zip(critic1_grads, self.critic1.trainable_variables))
        
        critic2_grads = tape.gradient(critic2_loss, self.critic2.trainable_variables)
        self.critic2_optimizer.apply_gradients(zip(critic2_grads, self.critic2.trainable_variables))
        
        # Update actor
        with tf.GradientTape() as tape:
            mean, log_std = self.actor(states, training=True)
            std = tf.exp(log_std)
            normal = tf.random.normal(shape=mean.shape)
            actions_pred = mean + std * normal
            actions_pred_tanh = tf.tanh(actions_pred)
            
            # Scale actions
            actions_pred_scaled = actions_pred_tanh * (self.action_high - self.action_low) / 2 + \
                                (self.action_high + self.action_low) / 2
            
            # Compute log probability
            log_prob = -0.5 * tf.reduce_sum(tf.square((actions_pred - mean) / std) + 
                                          2 * log_std + np.log(2 * np.pi), axis=1, keepdims=True)
            
            # Adjust for tanh squashing
            log_prob -= tf.reduce_sum(tf.math.log(1 - actions_pred_tanh**2 + 1e-6), axis=1, keepdims=True)
            
            q1 = self.critic1([states, actions_pred_scaled], training=True)
            actor_loss = tf.reduce_mean(self.alpha * log_prob - q1)
        
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        # Update temperature
        if self.auto_entropy_tuning:
            with tf.GradientTape() as tape:
                alpha_loss = -tf.reduce_mean(self.log_alpha * (log_prob + self.target_entropy))
            
            alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
            self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))
            self.alpha = tf.exp(self.log_alpha)
        
        # Update target networks
        self.update_target_networks()
        
        del tape
    
    def train(self, env, episodes=1000, max_steps=500):
        scores = []
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                action = self.sample_action(state)
                next_state, reward, done, _ = env.step(action)
                
                self.remember(state, action, reward, next_state, done)
                self.replay()
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            scores.append(total_reward)
            
            if episode % 100 == 0:
                avg_score = np.mean(scores[-100:]) if scores else 0
                print(f"Episode {episode}, Avg Score: {avg_score:.2f}, Alpha: {self.alpha.numpy():.3f}")
    
    def save_models(self, actor_path, critic1_path, critic2_path):
        self.actor.save(actor_path)
        self.critic1.save(critic1_path)
        self.critic2.save(critic2_path)
    
    def load_models(self, actor_path, critic1_path, critic2_path):
        self.actor = tf.keras.models.load_model(actor_path)
        self.critic1 = tf.keras.models.load_model(critic1_path)
        self.critic2 = tf.keras.models.load_model(critic2_path)
        self.update_target_networks(tau=1.0)