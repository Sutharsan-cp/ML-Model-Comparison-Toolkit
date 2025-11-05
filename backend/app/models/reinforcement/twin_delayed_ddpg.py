import numpy as np
import random
import tensorflow as tf
from collections import deque

class TD3:
    def __init__(self, state_size, action_size, action_low, action_high,
                 actor_lr=0.0001, critic_lr=0.001, discount_factor=0.99,
                 replay_memory_size=10000, batch_size=64, tau=0.005,
                 exploration_noise=0.1, target_policy_noise=0.2, 
                 target_noise_clip=0.5, policy_delay=2):
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.lr_actor = actor_lr
        self.lr_critic = critic_lr
        self.gamma = discount_factor
        self.batch_size = batch_size
        self.tau = tau
        self.exploration_noise = exploration_noise
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
        self.policy_delay = policy_delay
        
        # Replay memory
        self.memory = deque(maxlen=replay_memory_size)
        
        # Actor and Critic networks
        self.actor = self._build_actor()
        self.critic1 = self._build_critic()
        self.critic2 = self._build_critic()
        
        # Target networks
        self.target_actor = self._build_actor()
        self.target_critic1 = self._build_critic()
        self.target_critic2 = self._build_critic()
        
        # Initialize target networks
        self.update_target_networks(tau=1.0)
        
        # Optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic1_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        self.critic2_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        
        self.training_step = 0
    
    def _build_actor(self):
        inputs = tf.keras.layers.Input(shape=(self.state_size,))
        x = tf.keras.layers.Dense(400, activation='relu')(inputs)
        x = tf.keras.layers.Dense(300, activation='relu')(x)
        outputs = tf.keras.layers.Dense(self.action_size, activation='tanh')(x)
        
        # Scale outputs to action space
        outputs = tf.keras.layers.Lambda(lambda x: x * (self.action_high - self.action_low) / 2 + 
                                        (self.action_high + self.action_low) / 2)(outputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def _build_critic(self):
        # State input
        state_input = tf.keras.layers.Input(shape=(self.state_size,))
        state_out = tf.keras.layers.Dense(400, activation='relu')(state_input)
        state_out = tf.keras.layers.Dense(300, activation='relu')(state_out)
        
        # Action input
        action_input = tf.keras.layers.Input(shape=(self.action_size,))
        action_out = tf.keras.layers.Dense(300, activation='relu')(action_input)
        
        # Combine state and action
        concat = tf.keras.layers.Concatenate()([state_out, action_out])
        x = tf.keras.layers.Dense(300, activation='relu')(concat)
        outputs = tf.keras.layers.Dense(1, activation='linear')(x)
        
        return tf.keras.Model(inputs=[state_input, action_input], outputs=outputs)
    
    def update_target_networks(self, tau=None):
        tau = tau or self.tau
        
        # Update target networks using soft updates
        networks = [
            (self.actor, self.target_actor),
            (self.critic1, self.target_critic1),
            (self.critic2, self.target_critic2)
        ]
        
        for online_net, target_net in networks:
            online_weights = online_net.get_weights()
            target_weights = target_net.get_weights()
            for i in range(len(online_weights)):
                target_weights[i] = tau * online_weights[i] + (1 - tau) * target_weights[i]
            target_net.set_weights(target_weights)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def choose_action(self, state, training=True):
        state = np.reshape(state, [1, self.state_size])
        action = self.actor.predict(state, verbose=0)[0]
        
        if training:
            # Add exploration noise
            noise = np.random.normal(0, self.exploration_noise, size=self.action_size)
            action = np.clip(action + noise, self.action_low, self.action_high)
        
        return action
    
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
            # Target policy smoothing
            target_actions = self.target_actor(next_states, training=True)
            noise = tf.random.normal(shape=target_actions.shape, stddev=self.target_policy_noise)
            noise = tf.clip_by_value(noise, -self.target_noise_clip, self.target_noise_clip)
            target_actions = tf.clip_by_value(target_actions + noise, self.action_low, self.action_high)
            
            # Target Q values
            target_q1 = self.target_critic1([next_states, target_actions], training=True)
            target_q2 = self.target_critic2([next_states, target_actions], training=True)
            target_q = tf.minimum(target_q1, target_q2)
            target_q = rewards + self.gamma * target_q * (1 - dones)
            
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
        
        del tape
        
        # Delayed policy update
        self.training_step += 1
        if self.training_step % self.policy_delay == 0:
            with tf.GradientTape() as tape:
                actions_pred = self.actor(states, training=True)
                actor_loss = -tf.reduce_mean(self.critic1([states, actions_pred], training=True))
            
            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
            
            # Update target networks
            self.update_target_networks()
    
    def train(self, env, episodes=1000, max_steps=500):
        scores = []
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                action = self.choose_action(state)
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
                print(f"Episode {episode}, Avg Score: {avg_score:.2f}")
    
    def save_models(self, actor_path, critic1_path, critic2_path):
        self.actor.save(actor_path)
        self.critic1.save(critic1_path)
        self.critic2.save(critic2_path)
    
    def load_models(self, actor_path, critic1_path, critic2_path):
        self.actor = tf.keras.models.load_model(actor_path)
        self.critic1 = tf.keras.models.load_model(critic1_path)
        self.critic2 = tf.keras.models.load_model(critic2_path)
        self.update_target_networks(tau=1.0)