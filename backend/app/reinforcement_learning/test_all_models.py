"""
Comprehensive test for all RL models
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

# Continuous State Environment (for DQN, etc.)
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

print("Testing All Reinforcement Learning Models")
print("="*60)

registry = ReinforcementLearningRegistry()
trainer = ReinforcementLearningTrainer()

# Test Value-Based Models (Discrete)
print("\n📊 VALUE-BASED MODELS (Discrete State)")
print("-"*60)

discrete_env = SimpleGridWorld(size=5)
value_based_models = registry.get_value_based_models()

for model_name in ['qlearning', 'sarsa', 'montecarlo']:
    if model_name in value_based_models:
        print(f"\nTesting {value_based_models[model_name]['name']}...")
        try:
            result = trainer.train_model(
                model_name,
                discrete_env,
                'value_based',
                episodes=30,
                max_steps=50,
                verbose=False
            )
            
            if result['training_successful']:
                print(f"   ✓ Success! Avg Reward: {result['avg_reward']:.2f}")
            else:
                print(f"   ✗ Failed: {result['error'][:100]}")
        except Exception as e:
            print(f"   ✗ Exception: {str(e)[:100]}")

# Test Value-Based Models (Continuous State)
print("\n\n📊 VALUE-BASED MODELS (Continuous State)")
print("-"*60)

continuous_env = ContinuousGridWorld(size=5)

for model_name in ['dqn', 'doubledqn']:
    if model_name in value_based_models:
        print(f"\nTesting {value_based_models[model_name]['name']}...")
        try:
            result = trainer.train_model(
                model_name,
                continuous_env,
                'value_based',
                hyperparameters={
                    'learning_rate': 0.001,
                    'discount_factor': 0.99,
                    'batch_size': 16,
                    'target_update_freq': 5
                },
                episodes=20,
                max_steps=50,
                verbose=False
            )
            
            if result['training_successful']:
                print(f"   ✓ Success! Avg Reward: {result['avg_reward']:.2f}")
            else:
                print(f"   ✗ Failed: {result['error'][:100]}")
        except Exception as e:
            print(f"   ✗ Exception: {str(e)[:100]}")

# Test Policy-Based Models
print("\n\n🎮 POLICY-BASED MODELS")
print("-"*60)

policy_based_models = registry.get_policy_based_models()

for model_name in policy_based_models.keys():
    print(f"\nTesting {policy_based_models[model_name]['name']}...")
    try:
        result = trainer.train_model(
            model_name,
            continuous_env,
            'policy_based',
            episodes=20,
            max_steps=50,
            verbose=False
        )
        
        if result['training_successful']:
            print(f"   ✓ Success! Avg Reward: {result['avg_reward']:.2f}")
        else:
            print(f"   ✗ Failed: {result['error'][:100]}")
    except Exception as e:
        print(f"   ✗ Exception: {str(e)[:100]}")

# Test Actor-Critic Models
print("\n\n🤝 ACTOR-CRITIC MODELS")
print("-"*60)

actor_critic_models = registry.get_actor_critic_models()

for model_name in actor_critic_models.keys():
    print(f"\nTesting {actor_critic_models[model_name]['name']}...")
    try:
        result = trainer.train_model(
            model_name,
            continuous_env,
            'actor_critic',
            episodes=20,
            max_steps=50,
            verbose=False
        )
        
        if result['training_successful']:
            print(f"   ✓ Success! Avg Reward: {result['avg_reward']:.2f}")
        else:
            print(f"   ✗ Failed: {result['error'][:100]}")
    except Exception as e:
        print(f"   ✗ Exception: {str(e)[:100]}")

# Summary
print("\n\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

summary = trainer.get_training_summary()
print(f"\nTotal Models Tested: {len(summary)}")
print(f"Successful: {summary['Training_Successful'].sum()}")
print(f"Failed: {(~summary['Training_Successful']).sum()}")

print("\nDetailed Results:")
print(summary[['Model', 'Training_Successful', 'Avg_Reward', 'Error']].to_string())

print("\n✅ Testing completed!")
