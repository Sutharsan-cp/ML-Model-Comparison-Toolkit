import numpy as np
import tensorflow as tf

class PPO:
    def __init__(self, state_size, action_size, actor_lr=0.0001, critic_lr=0.0005,
                 discount_factor=0.99, gae_lambda=0.95, clip_ratio=0.2, 
                 value_coef=0.5, entropy_coef=0.01, epochs=10, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = discount_factor
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Build actor and critic networks
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        
        self.states = []
        self.actions = []
        self.old_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
    
    def _build_actor(self):
        inputs = tf.keras.layers.Input(shape=(self.state_size,))
        x = tf.keras.layers.Dense(64, activation='tanh')(inputs)
        x = tf.keras.layers.Dense(64, activation='tanh')(x)
        outputs = tf.keras.layers.Dense(self.action_size, activation='softmax')(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def _build_critic(self):
        inputs = tf.keras.layers.Input(shape=(self.state_size,))
        x = tf.keras.layers.Dense(64, activation='tanh')(inputs)
        x = tf.keras.layers.Dense(64, activation='tanh')(x)
        outputs = tf.keras.layers.Dense(1, activation='linear')(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def choose_action(self, state):
        state = np.reshape(state, [1, self.state_size])
        action_probs = self.actor.predict(state, verbose=0)[0]
        value = self.critic.predict(state, verbose=0)[0, 0]
        action = np.random.choice(self.action_size, p=action_probs)
        
        # Store for training
        self.states.append(state[0])
        self.actions.append(action)
        self.old_probs.append(action_probs[action])
        self.values.append(value)
        
        return action
    
    def store_transition(self, state, action, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)
    
    def compute_advantages(self, next_value):
        """Compute Generalized Advantage Estimation (GAE)"""
        values = np.array(self.values + [next_value])
        advantages = np.zeros_like(self.rewards)
        gae = 0
        
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * values[t+1] * (1 - self.dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values[:-1]
        return advantages, returns
    
    def learn(self, next_state):
        if len(self.states) == 0:
            return
        
        # Compute next value
        next_state = np.reshape(next_state, [1, self.state_size])
        next_value = self.critic.predict(next_state, verbose=0)[0, 0]
        
        # Compute advantages and returns
        advantages, returns = self.compute_advantages(next_value)
        
        # Convert to arrays
        states = np.array(self.states)
        actions = np.array(self.actions)
        old_probs = np.array(self.old_probs)
        returns = np.array(returns)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # Train for multiple epochs
        for epoch in range(self.epochs):
            # Shuffle data
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_probs = old_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Update actor
                with tf.GradientTape() as tape:
                    action_probs = self.actor(batch_states, training=True)
                    action_mask = tf.one_hot(batch_actions, self.action_size)
                    new_probs = tf.reduce_sum(action_probs * action_mask, axis=1)
                    
                    ratio = new_probs / (batch_old_probs + 1e-8)
                    surr1 = ratio * batch_advantages
                    surr2 = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                    
                    actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
                    
                    # Entropy bonus
                    entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-8), axis=1)
                    entropy_loss = -tf.reduce_mean(entropy)
                    
                    total_actor_loss = actor_loss + self.entropy_coef * entropy_loss
                
                actor_grads = tape.gradient(total_actor_loss, self.actor.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
                
                # Update critic
                with tf.GradientTape() as tape:
                    current_values = self.critic(batch_states, training=True)
                    critic_loss = tf.keras.losses.MSE(batch_returns, current_values)
                    total_critic_loss = self.value_coef * critic_loss
                
                critic_grads = tape.gradient(total_critic_loss, self.critic.trainable_variables)
                self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        # Clear memory
        self.states = []
        self.actions = []
        self.old_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
    
    def train(self, env, episodes=1000, max_steps=500):
        scores = []
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                action = self.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                
                self.store_transition(state, action, reward, done)
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            # Learn from the episode
            self.learn(next_state)
            
            scores.append(total_reward)
            
            if episode % 100 == 0:
                avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
                print(f"Episode {episode}, Avg Score: {avg_score:.2f}")
    
    def save_models(self, actor_path, critic_path):
        self.actor.save(actor_path)
        self.critic.save(critic_path)
    
    def load_models(self, actor_path, critic_path):
        self.actor = tf.keras.models.load_model(actor_path)
        self.critic = tf.keras.models.load_model(critic_path)