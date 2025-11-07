"""
Test working RL models
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from reinforcement_learning import ReinforcementLearningRegistry, ReinforcementLearningTrainer
import numpy as np

# Simple Grid World Environment
class SimpleGridWorld:
    def __init__(self, size=5):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4
        self.observation_space = type('obj', (object,), {'n': self.n_states})()
        self.action_space = type('obj', (object,), {'n': self.n_actions})()
        self.reset()
    
    def reset(self):
        self.agent_pos = [0, 0]
        self.goal_pos = [self.size-1, self.size-1]
        return self._get_state()
    
    def _get_state(self):
        return self.agent_pos[0] * self.size + self.agent_pos[1]
    
    def step(self, action):
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

# Continuous State Environment
class ContinuousGridWorld:
    def __init__(self, size=5):
        self.size = size
        self.n_actions = 4
        self.observation_space = type('obj', (object,), {'shape': (2,)})()
        self.action_space = type('obj', (object,), {'n': self.n_actions})()
        self.reset()
    
    def reset(self):
        self.agent_pos = np.array([0.0, 0.0])
        self.goal_pos = np.array([float(self.size-1), float(self.size-1)])
        return self.agent_pos.copy()
    
    def step(self, action):
        if action == 0 and self.agent_pos[0] > 0:
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[0] < self.size - 1:
            self.agent_pos[0] += 1
        elif action == 2 and self.agent_pos[1] > 0:
            self.agent_pos[1] -= 1
        elif action == 3 and self.agent_pos[1] < self.size - 1:
            self.agent_pos[1] += 1
        
        done = np.array_equal(self.agent_pos, self.goal_pos)
        reward = 10.0 if done else -0.1
        
        return self.agent_pos.copy(), reward, done, {}

print("Testing Working Reinforcement Learning Models")
print("="*60)

registry = ReinforcementLearningRegistry()
trainer = ReinforcementLearningTrainer()

discrete_env = SimpleGridWorld(size=5)
continuous_env = ContinuousGridWorld(size=5)

# Test models
test_models = [
    ('qlearning', 'value_based', discrete_env, {}),
    ('sarsa', 'value_based', discrete_env, {}),
    ('montecarlo', 'value_based', discrete_env, {}),
    ('dqn', 'value_based', continuous_env, {'learning_rate': 0.001, 'batch_size': 16}),
    ('doubledqn', 'value_based', continuous_env, {'learning_rate': 0.001, 'batch_size': 16}),
    ('policygradient', 'policy_based', continuous_env, {}),
    ('reinforce', 'policy_based', continuous_env, {}),
    ('actorcritic', 'actor_critic', continuous_env, {}),
]

results = []

for model_name, category, env, hyperparams in test_models:
    print(f"\nTesting {model_name}...")
    try:
        result = trainer.train_model(
            model_name,
            env,
            category,
            hyperparameters=hyperparams,
            episodes=30,
            max_steps=50,
            verbose=False
        )
        
        if result['training_successful']:
            print(f"   ✓ Success! Avg Reward: {result['avg_reward']:.2f}")
            results.append((model_name, '✓', result['avg_reward']))
        else:
            print(f"   ✗ Failed: {result['error'][:80]}")
            results.append((model_name, '✗', 'Failed'))
    except Exception as e:
        print(f"   ✗ Exception: {str(e)[:80]}")
        results.append((model_name, '✗', 'Exception'))

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

for model_name, status, reward in results:
    print(f"{model_name:20s} {status:3s} {reward}")

successful = sum(1 for _, status, _ in results if status == '✓')
print(f"\nSuccessful: {successful}/{len(results)}")

print("\n✅ Testing completed!")
