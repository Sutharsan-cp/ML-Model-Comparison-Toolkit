"""
Quick test for reinforcement learning module
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from reinforcement_learning import ReinforcementLearningRegistry, ReinforcementLearningTrainer

# Simple Grid World Environment
class SimpleGridWorld:
    def __init__(self, size=5):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4  # up, down, left, right
        self.reset()
    
    def reset(self):
        self.agent_pos = [0, 0]
        self.goal_pos = [self.size-1, self.size-1]
        return self._get_state()
    
    def _get_state(self):
        return self.agent_pos[0] * self.size + self.agent_pos[1]
    
    def step(self, action):
        # Actions: 0=up, 1=down, 2=left, 3=right
        if action == 0 and self.agent_pos[0] > 0:
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[0] < self.size - 1:
            self.agent_pos[0] += 1
        elif action == 2 and self.agent_pos[1] > 0:
            self.agent_pos[1] -= 1
        elif action == 3 and self.agent_pos[1] < self.size - 1:
            self.agent_pos[1] += 1
        
        done = (self.agent_pos == self.goal_pos)
        reward = 10.0 if done else -0.1
        
        return self._get_state(), reward, done, {}

# Test
print("Testing Reinforcement Learning Module")
print("="*50)

registry = ReinforcementLearningRegistry()
trainer = ReinforcementLearningTrainer()

# Create environment
env = SimpleGridWorld(size=5)

print(f"\nEnvironment: {env.size}x{env.size} Grid World")
print(f"States: {env.n_states}, Actions: {env.n_actions}")

# Test Q-Learning
print("\n1. Testing Q-Learning...")
try:
    result = trainer.train_model(
        'qlearning',
        env,
        'value_based',
        hyperparameters={
            'learning_rate': 0.1,
            'discount_factor': 0.99,
            'exploration_rate': 1.0,
            'exploration_decay': 0.995,
            'min_exploration': 0.01
        },
        episodes=50,
        max_steps=100,
        verbose=False
    )
    
    if result['training_successful']:
        print(f"   ✓ Q-Learning trained successfully!")
        print(f"   Average Reward: {result['avg_reward']:.2f}")
        print(f"   Max Reward: {result['max_reward']:.2f}")
    else:
        print(f"   ✗ Q-Learning failed: {result['error']}")
except Exception as e:
    print(f"   ✗ Q-Learning exception: {str(e)}")

# Test SARSA
print("\n2. Testing SARSA...")
try:
    result = trainer.train_model(
        'sarsa',
        env,
        'value_based',
        hyperparameters={
            'learning_rate': 0.1,
            'discount_factor': 0.99,
            'exploration_rate': 1.0,
            'exploration_decay': 0.995
        },
        episodes=50,
        max_steps=100,
        verbose=False
    )
    
    if result['training_successful']:
        print(f"   ✓ SARSA trained successfully!")
        print(f"   Average Reward: {result['avg_reward']:.2f}")
    else:
        print(f"   ✗ SARSA failed: {result['error']}")
except Exception as e:
    print(f"   ✗ SARSA exception: {str(e)}")

# Test Monte Carlo
print("\n3. Testing Monte Carlo...")
try:
    result = trainer.train_model(
        'montecarlo',
        env,
        'value_based',
        hyperparameters={
            'discount_factor': 0.99,
            'exploration_rate': 0.2
        },
        episodes=50,
        max_steps=100,
        verbose=False
    )
    
    if result['training_successful']:
        print(f"   ✓ Monte Carlo trained successfully!")
        print(f"   Average Reward: {result['avg_reward']:.2f}")
    else:
        print(f"   ✗ Monte Carlo failed: {result['error']}")
except Exception as e:
    print(f"   ✗ Monte Carlo exception: {str(e)}")

print("\n" + "="*50)
print("Test Summary:")
summary = trainer.get_training_summary()
print(summary[['Model', 'Training_Successful', 'Avg_Reward']])

print("\nTest completed!")
