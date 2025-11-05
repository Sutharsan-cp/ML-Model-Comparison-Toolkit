import numpy as np
import tensorflow as tf

class AdvantageActorCritic:
    def __init__(self, state_size, action_size, actor_lr=0.001, critic_lr=0.005, 
                 discount_factor=0.99, n_steps=5):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = discount_factor
        self.n_steps = n_steps
        
        # Build actor and critic networks
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        
        self.states = []
        self.actions = []
        self.rewards = []
    
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
    
    def choose_action(self, state):
        state = np.reshape(state, [1, self.state_size])
        probabilities = self.actor.predict(state, verbose=0)[0]
        action = np.random.choice(self.action_size, p=probabilities)
        return action
    
    def store_transition(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def compute_n_step_returns(self, next_state):
        """Compute n-step returns for advantage estimation"""
        if len(self.rewards) < self.n_steps:
            return self.compute_advantages()
        
        states = np.vstack(self.states)
        values = self.critic.predict(states, verbose=0).flatten()
        
        advantages = []
        returns = []
        
        # Compute n-step returns
        for t in range(len(self.rewards)):
            n_step_return = 0
            for k in range(self.n_steps):
                if t + k < len(self.rewards):
                    n_step_return += (self.gamma ** k) * self.rewards[t + k]
                else:
                    break
            
            if t + self.n_steps < len(self.states):
                next_value = self.critic.predict(
                    np.reshape(self.states[t + self.n_steps], [1, self.state_size]), verbose=0
                )[0, 0]
                n_step_return += (self.gamma ** self.n_steps) * next_value
            
            advantage = n_step_return - values[t]
            advantages.append(advantage)
            returns.append(n_step_return)
        
        return advantages, returns
    
    def learn(self):
        if len(self.states) == 0:
            return
        
        advantages, returns = self.compute_n_step_returns()
        
        states = np.vstack(self.states)
        actions = np.array(self.actions)
        returns = np.array(returns)
        advantages = np.array(advantages)
        
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # Update critic
        with tf.GradientTape() as tape:
            current_values = self.critic(states, training=True)
            critic_loss = tf.keras.losses.MSE(returns, current_values)
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        # Update actor
        with tf.GradientTape() as tape:
            action_probs = self.actor(states, training=True)
            action_mask = tf.one_hot(actions, self.action_size)
            chosen_action_probs = tf.reduce_sum(action_probs * action_mask, axis=1)
            actor_loss = -tf.reduce_mean(tf.math.log(chosen_action_probs + 1e-8) * advantages)
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        # Clear memory
        self.states = []
        self.actions = []
        self.rewards = []
    
    def train(self, env, episodes=1000, max_steps=500):
        scores = []
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                action = self.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                
                self.store_transition(state, action, reward)
                
                state = next_state
                total_reward += reward
                
                # Learn every n steps or at the end of episode
                if len(self.states) >= self.n_steps or done:
                    self.learn()
                
                if done:
                    break
            
            # Learn from any remaining transitions
            if len(self.states) > 0:
                self.learn()
            
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