"""
Test script for reinforcement learning models
"""

import sys
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import our RL module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from reinforcement_learning import ReinforcementLearningRegistry, ReinforcementLearningTrainer

# Simple environment classes for testing
class SimpleGridWorld:
    """Simple grid world environment for testing"""
    
    def __init__(self, size=4):
        self.size = size
        self.state_dim = size * size
        self.action_dim = 4  # up, down, left, right
        self.reset()
    
    def reset(self):
        self.agent_pos = [0, 0]
        self.goal_pos = [self.size-1, self.size-1]
        return self._get_state()
    
    def _get_state(self):
        return self.agent_pos[0] * self.size + self.agent_pos[1]
    
    def step(self, action):
        # Actions: 0=up, 1=down, 2=left, 3=right
        old_pos = self.agent_pos.copy()
        
        if action == 0 and self.agent_pos[0] > 0:  # up
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[0] < self.size-1:  # down
            self.agent_pos[0] += 1
        elif action == 2 and self.agent_pos[1] > 0:  # left
            self.agent_pos[1] -= 1
        elif action == 3 and self.agent_pos[1] < self.size-1:  # right
            self.agent_pos[1] += 1
        
        # Calculate reward
        if self.agent_pos == self.goal_pos:
            reward = 10
            done = True
        else:
            reward = -0.1  # Small negative reward for each step
            done = False
        
        return self._get_state(), reward, done, {}

class SimpleContinuousEnv:
    """Simple continuous environment for testing"""
    
    def __init__(self):
        self.state_dim = 2
        self.action_dim = 1
        self.reset()
    
    def reset(self):
        self.state = np.random.uniform(-1, 1, 2)
        return self.state.copy()
    
    def step(self, action):
        # Simple dynamics: move towards origin
        action = np.clip(action, -1, 1)
        self.state[0] += 0.1 * action
        self.state[1] += 0.1 * (0.5 - self.state[1])
        
        # Reward is negative distance from origin
        reward = -np.linalg.norm(self.state)
        
        # Episode ends when close to origin or after many steps
        done = np.linalg.norm(self.state) < 0.1
        
        return self.state.copy(), reward, done, {}

def test_value_based_models():
    """Test value-based RL models"""
    print("Testing Value-Based RL Models")
    print("=" * 50)
    
    # Create simple environment
    env = SimpleGridWorld(size=3)
    
    # Initialize registry and trainer
    registry = ReinforcementLearningRegistry()
    trainer = ReinforcementLearningTrainer()
    
    # Get available value-based models
    value_based_models = registry.get_value_based_models()
    print(f"Available value-based models: {len(value_based_models)}")
    print(f"Test environment: {env.size}x{env.size} GridWorld")
    
    # Test value-based models
    test_models = list(value_based_models.keys())
    
    if test_models:
        print(f"Testing models: {test_models}")
        
        for model_name in test_models:
            print(f"\nTraining {model_name}...")
            try:
                # Use simple hyperparameters for testing
                hyperparams = {}
                if model_name in ['q_learning', 'sarsa']:
                    hyperparams = {
                        'n_states': env.state_dim,
                        'n_actions': env.action_dim,
                        'learning_rate': 0.1,
                        'exploration_rate': 0.1
                    }
                elif model_name == 'monte_carlo':
                    hyperparams = {
                        'n_states': env.state_dim,
                        'n_actions': env.action_dim,
                        'discount_factor': 0.9,
                        'exploration_rate': 0.1
                    }
                elif 'dqn' in model_name:
                    hyperparams = {
                        'state_size': env.state_dim,
                        'action_size': env.action_dim,
                        'learning_rate': 0.001,
                        'exploration_rate': 0.1,
                        'batch_size': 16
                    }
                
                result = trainer.train_model(
                    model_name, env, 'value_based', hyperparams, episodes=50
                )
                
                if result['training_successful']:
                    final_reward = result.get('final_reward', 'N/A')
                    avg_reward = result.get('average_reward', 'N/A')
                    best_reward = result.get('best_reward', 'N/A')
                    
                    if isinstance(final_reward, float):
                        final_reward = f"{final_reward:.2f}"
                    if isinstance(avg_reward, float):
                        avg_reward = f"{avg_reward:.2f}"
                    if isinstance(best_reward, float):
                        best_reward = f"{best_reward:.2f}"
                    
                    print(f"✅ {result['model_display_name']}: "
                          f"Final={final_reward}, Avg={avg_reward}, Best={best_reward}")
                else:
                    print(f"❌ {result['model_display_name']}: {result['error']}")
            
            except Exception as e:
                print(f"❌ {model_name}: {str(e)}")
    else:
        print("No value-based models available")
    
    print()

def test_policy_based_models():
    """Test policy-based RL models"""
    print("Testing Policy-Based RL Models")
    print("=" * 50)
    
    # Create simple environment
    env = SimpleContinuousEnv()
    
    # Initialize registry and trainer
    registry = ReinforcementLearningRegistry()
    trainer = ReinforcementLearningTrainer()
    
    # Get available policy-based models
    policy_based_models = registry.get_policy_based_models()
    print(f"Available policy-based models: {len(policy_based_models)}")
    print(f"Test environment: Continuous 2D environment")
    
    # Test policy-based models
    test_models = list(policy_based_models.keys())
    
    if test_models:
        print(f"Testing models: {test_models}")
        
        for model_name in test_models:
            print(f"\nTraining {model_name}...")
            try:
                # Use simple hyperparameters for testing
                hyperparams = {
                    'state_size': env.state_dim,
                    'action_size': env.action_dim,
                    'learning_rate': 0.001
                }
                
                result = trainer.train_model(
                    model_name, env, 'policy_based', hyperparams, episodes=50
                )
                
                if result['training_successful']:
                    final_reward = result.get('final_reward', 'N/A')
                    avg_reward = result.get('average_reward', 'N/A')
                    
                    if isinstance(final_reward, float):
                        final_reward = f"{final_reward:.2f}"
                    if isinstance(avg_reward, float):
                        avg_reward = f"{avg_reward:.2f}"
                    
                    print(f"✅ {result['model_display_name']}: "
                          f"Final={final_reward}, Avg={avg_reward}")
                else:
                    print(f"❌ {result['model_display_name']}: {result['error']}")
            
            except Exception as e:
                print(f"❌ {model_name}: {str(e)}")
    else:
        print("No policy-based models available")
    
    print()

def test_actor_critic_models():
    """Test actor-critic RL models"""
    print("Testing Actor-Critic RL Models")
    print("=" * 50)
    
    # Create simple environment
    env = SimpleContinuousEnv()
    
    # Initialize registry and trainer
    registry = ReinforcementLearningRegistry()
    trainer = ReinforcementLearningTrainer()
    
    # Get available actor-critic models
    actor_critic_models = registry.get_actor_critic_models()
    print(f"Available actor-critic models: {len(actor_critic_models)}")
    print(f"Test environment: Continuous 2D environment")
    
    # Test actor-critic models
    test_models = list(actor_critic_models.keys())
    
    if test_models:
        print(f"Testing models: {test_models}")
        
        for model_name in test_models:
            print(f"\nTraining {model_name}...")
            try:
                # Use simple hyperparameters for testing
                hyperparams = {
                    'state_size': env.state_dim,
                    'action_size': env.action_dim,
                    'learning_rate': 0.001
                }
                
                result = trainer.train_model(
                    model_name, env, 'actor_critic', hyperparams, episodes=30
                )
                
                if result['training_successful']:
                    final_reward = result.get('final_reward', 'N/A')
                    avg_reward = result.get('average_reward', 'N/A')
                    
                    if isinstance(final_reward, float):
                        final_reward = f"{final_reward:.2f}"
                    if isinstance(avg_reward, float):
                        avg_reward = f"{avg_reward:.2f}"
                    
                    print(f"✅ {result['model_display_name']}: "
                          f"Final={final_reward}, Avg={avg_reward}")
                else:
                    print(f"❌ {result['model_display_name']}: {result['error']}")
            
            except Exception as e:
                print(f"❌ {model_name}: {str(e)}")
    else:
        print("No actor-critic models available")
    
    print()

def test_model_comparison():
    """Test model comparison functionality"""
    print("Testing Model Comparison")
    print("=" * 50)
    
    # Create simple environment
    env = SimpleGridWorld(size=3)
    
    # Initialize trainer
    trainer = ReinforcementLearningTrainer()
    
    # Train a few models for comparison
    test_models = ['q_learning', 'sarsa']
    results = []
    
    for model_name in test_models:
        try:
            hyperparams = {
                'n_states': env.state_dim,
                'n_actions': env.action_dim,
                'learning_rate': 0.1,
                'exploration_rate': 0.1
            }
            
            result = trainer.train_model(model_name, env, 'value_based', hyperparams, episodes=30)
            results.append(result)
        except Exception as e:
            print(f"Failed to train {model_name}: {e}")
    
    # Generate comparison report
    from reinforcement_learning import create_model_comparison_report
    
    try:
        comparison_df = create_model_comparison_report(results)
        print("✅ Comparison report generated successfully")
        print(f"Report shape: {comparison_df.shape}")
        if not comparison_df.empty:
            print("\nTop performing models:")
            print(comparison_df.head())
    except Exception as e:
        print(f"❌ Failed to generate comparison report: {e}")
    
    print()

def run_all_tests():
    """Run all RL tests"""
    print("Reinforcement Learning Models Test Suite")
    print("=" * 60)
    print()
    
    test_value_based_models()
    test_policy_based_models()
    test_actor_critic_models()
    test_model_comparison()
    
    print("=" * 60)
    print("All tests completed!")

if __name__ == "__main__":
    run_all_tests()