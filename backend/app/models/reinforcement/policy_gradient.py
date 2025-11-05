import numpy as np
import tensorflow as tf

class PolicyGradient:
    def __init__(self, state_size, action_size, learning_rate=0.01, discount_factor=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        
        self.states = []
        self.actions = []
        self.rewards = []
        
        self.model = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
    
    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='softmax')
        ])
        return model
    
    def choose_action(self, state):
        state = np.reshape(state, [1, self.state_size])
        probabilities = self.model.predict(state, verbose=0)[0]
        action = np.random.choice(self.action_size, p=probabilities)
        return action
    
    def store_transition(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def learn(self):
        # Calculate discounted rewards
        discounted_rewards = np.zeros_like(self.rewards)
        running_add = 0
        for t in reversed(range(len(self.rewards))):
            running_add = running_add * self.gamma + self.rewards[t]
            discounted_rewards[t] = running_add
        
        # Normalize rewards
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards) if np.std(discounted_rewards) > 0 else 1
        
        # Convert to tensors
        states = np.vstack(self.states)
        actions = np.array(self.actions)
        
        with tf.GradientTape() as tape:
            # Get action probabilities
            action_probs = self.model(states, training=True)
            
            # Calculate loss
            action_mask = tf.one_hot(actions, self.action_size)
            chosen_action_probs = tf.reduce_sum(action_probs * action_mask, axis=1)
            
            # Policy gradient loss
            loss = -tf.reduce_mean(tf.math.log(chosen_action_probs + 1e-8) * discounted_rewards)
        
        # Apply gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
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
                
                if done:
                    break
            
            # Learn from the episode
            self.learn()
            
            scores.append(total_reward)
            
            if episode % 100 == 0:
                avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
                print(f"Episode {episode}, Avg Score: {avg_score:.2f}")
    
    def save_model(self, filepath):
        self.model.save(filepath)
    
    def load_model(self, filepath):
        self.model = tf.keras.models.load_model(filepath)