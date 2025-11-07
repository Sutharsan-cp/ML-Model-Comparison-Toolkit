"""
Reinforcement Learning Module
Consolidates all reinforcement learning algorithms for the ML Comparison Toolkit
Uses actual models from backend/app/models/reinforcement directory
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# Add models path to sys.path
models_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'reinforcement')
if models_path not in sys.path:
    sys.path.append(models_path)

# Import dictionaries for different categories
value_based_imports = {}
policy_based_imports = {}
actor_critic_imports = {}

# Value-Based Methods
try:
    from q_learning import QLearning
    value_based_imports['qlearning'] = QLearning
except ImportError as e:
    print(f"Warning: Could not import QLearning: {e}")

try:
    from sarsa import SARSA
    value_based_imports['sarsa'] = SARSA
except ImportError as e:
    print(f"Warning: Could not import SARSA: {e}")

try:
    from monte_carlo import MonteCarlo
    value_based_imports['montecarlo'] = MonteCarlo
except ImportError as e:
    print(f"Warning: Could not import MonteCarlo: {e}")

try:
    from deep_q_network import DQN
    value_based_imports['dqn'] = DQN
except ImportError as e:
    print(f"Warning: Could not import DQN: {e}")

try:
    from double_dqn import DoubleDQN
    value_based_imports['doubledqn'] = DoubleDQN
except ImportError as e:
    print(f"Warning: Could not import DoubleDQN: {e}")

# Policy-Based Methods
try:
    from policy_gradient import PolicyGradient
    policy_based_imports['policygradient'] = PolicyGradient
except ImportError as e:
    print(f"Warning: Could not import PolicyGradient: {e}")

try:
    from reinforce import REINFORCE
    policy_based_imports['reinforce'] = REINFORCE
except ImportError as e:
    print(f"Warning: Could not import REINFORCE: {e}")

try:
    from proximal_policy_optimization import PPO
    policy_based_imports['ppo'] = PPO
except ImportError as e:
    print(f"Warning: Could not import PPO: {e}")

# Actor-Critic Methods
try:
    from actor_critic import ActorCritic
    actor_critic_imports['actorcritic'] = ActorCritic
except ImportError as e:
    print(f"Warning: Could not import ActorCritic: {e}")

try:
    from deep_deterministic_policy_gradient import DDPG
    actor_critic_imports['ddpg'] = DDPG
except ImportError as e:
    print(f"Warning: Could not import DDPG: {e}")

try:
    from soft_actor_critic import SAC
    actor_critic_imports['sac'] = SAC
except ImportError as e:
    print(f"Warning: Could not import SAC: {e}")


class ReinforcementLearningRegistry:
    """Registry for all reinforcement learning models"""
    
    def __init__(self):
        # Build value-based models dictionary
        self.value_based_models = {}
        
        # Define value-based model configurations
        value_based_configs = {
            'qlearning': {
                'name': 'Q-Learning',
                'type': 'value_based',
                'description': 'Tabular off-policy TD control algorithm',
                'environment_types': ['discrete_state', 'discrete_action'],
                'task_types': ['control', 'optimization'],
                'hyperparameters': {
                    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.5],
                    'discount_factor': [0.9, 0.95, 0.99],
                    'exploration_rate': [0.5, 0.7, 1.0],
                    'exploration_decay': [0.99, 0.995, 0.999],
                    'min_exploration': [0.01, 0.05, 0.1]
                }
            },
            'sarsa': {
                'name': 'SARSA',
                'type': 'value_based',
                'description': 'On-policy TD control algorithm',
                'environment_types': ['discrete_state', 'discrete_action'],
                'task_types': ['control', 'optimization'],
                'hyperparameters': {
                    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.5],
                    'discount_factor': [0.9, 0.95, 0.99],
                    'exploration_rate': [0.5, 0.7, 1.0],
                    'exploration_decay': [0.99, 0.995, 0.999]
                }
            },
            'montecarlo': {
                'name': 'Monte Carlo',
                'type': 'value_based',
                'description': 'Episode-based learning using complete returns',
                'environment_types': ['discrete_state', 'discrete_action', 'episodic'],
                'task_types': ['control', 'prediction'],
                'hyperparameters': {
                    'discount_factor': [0.9, 0.95, 0.99],
                    'exploration_rate': [0.1, 0.2, 0.3],
                    'first_visit': [True, False]
                }
            },
            'dqn': {
                'name': 'Deep Q-Network',
                'type': 'value_based',
                'description': 'Deep learning-based Q-learning with experience replay',
                'environment_types': ['continuous_state', 'discrete_action'],
                'task_types': ['control', 'optimization'],
                'hyperparameters': {
                    'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
                    'discount_factor': [0.95, 0.99, 0.995],
                    'exploration_rate': [1.0],
                    'exploration_decay': [0.995, 0.999],
                    'batch_size': [32, 64, 128],
                    'target_update_freq': [50, 100, 200]
                }
            },
            'doubledqn': {
                'name': 'Double DQN',
                'type': 'value_based',
                'description': 'DQN with double Q-learning to reduce overestimation',
                'environment_types': ['continuous_state', 'discrete_action'],
                'task_types': ['control', 'optimization'],
                'hyperparameters': {
                    'learning_rate': [0.0001, 0.0005, 0.001],
                    'discount_factor': [0.95, 0.99],
                    'batch_size': [32, 64, 128],
                    'target_update_freq': [50, 100, 200]
                }
            }
        }
        
        # Add successfully imported value-based models
        for model_key, model_class in value_based_imports.items():
            if model_key in value_based_configs:
                config = value_based_configs[model_key].copy()
                config['class'] = model_class
                self.value_based_models[model_key] = config       
 
        # Build policy-based models dictionary
        self.policy_based_models = {}
        
        # Define policy-based model configurations
        policy_based_configs = {
            'policygradient': {
                'name': 'Policy Gradient',
                'type': 'policy_based',
                'description': 'Direct policy optimization using gradient ascent',
                'environment_types': ['continuous_state', 'discrete_action'],
                'task_types': ['control', 'optimization'],
                'hyperparameters': {
                    'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
                    'discount_factor': [0.95, 0.99, 0.995]
                }
            },
            'reinforce': {
                'name': 'REINFORCE',
                'type': 'policy_based',
                'description': 'Monte Carlo policy gradient algorithm',
                'environment_types': ['continuous_state', 'discrete_action', 'episodic'],
                'task_types': ['control', 'optimization'],
                'hyperparameters': {
                    'learning_rate': [0.0001, 0.0005, 0.001],
                    'discount_factor': [0.95, 0.99],
                    'baseline': [True, False]
                }
            },
            'ppo': {
                'name': 'Proximal Policy Optimization',
                'type': 'policy_based',
                'description': 'Policy optimization with clipped surrogate objective',
                'environment_types': ['continuous_state', 'discrete_action', 'continuous_action'],
                'task_types': ['control', 'optimization'],
                'hyperparameters': {
                    'actor_lr': [0.0001, 0.0003, 0.0005],
                    'critic_lr': [0.0003, 0.0005, 0.001],
                    'discount_factor': [0.95, 0.99],
                    'gae_lambda': [0.9, 0.95, 0.98],
                    'clip_ratio': [0.1, 0.2, 0.3],
                    'epochs': [5, 10, 20],
                    'batch_size': [32, 64, 128]
                }
            }
        }
        
        # Add successfully imported policy-based models
        for model_key, model_class in policy_based_imports.items():
            if model_key in policy_based_configs:
                config = policy_based_configs[model_key].copy()
                config['class'] = model_class
                self.policy_based_models[model_key] = config
        
        # Build actor-critic models dictionary
        self.actor_critic_models = {}
        
        # Define actor-critic model configurations
        actor_critic_configs = {
            'actorcritic': {
                'name': 'Actor-Critic',
                'type': 'actor_critic',
                'description': 'Combines value-based and policy-based methods',
                'environment_types': ['continuous_state', 'discrete_action'],
                'task_types': ['control', 'optimization'],
                'hyperparameters': {
                    'actor_lr': [0.0001, 0.0005, 0.001],
                    'critic_lr': [0.0005, 0.001, 0.005],
                    'discount_factor': [0.95, 0.99]
                }
            },
            'ddpg': {
                'name': 'Deep Deterministic Policy Gradient',
                'type': 'actor_critic',
                'description': 'Actor-critic for continuous action spaces',
                'environment_types': ['continuous_state', 'continuous_action'],
                'task_types': ['control', 'optimization'],
                'hyperparameters': {
                    'actor_lr': [0.0001, 0.0005, 0.001],
                    'critic_lr': [0.0005, 0.001, 0.005],
                    'discount_factor': [0.95, 0.99],
                    'tau': [0.001, 0.005, 0.01],
                    'batch_size': [64, 128, 256]
                }
            },
            'sac': {
                'name': 'Soft Actor-Critic',
                'type': 'actor_critic',
                'description': 'Maximum entropy actor-critic for robust learning',
                'environment_types': ['continuous_state', 'continuous_action'],
                'task_types': ['control', 'optimization'],
                'hyperparameters': {
                    'actor_lr': [0.0001, 0.0003],
                    'critic_lr': [0.0003, 0.001],
                    'discount_factor': [0.99],
                    'tau': [0.005, 0.01],
                    'alpha': [0.1, 0.2, 0.3]
                }
            }
        }
        
        # Add successfully imported actor-critic models
        for model_key, model_class in actor_critic_imports.items():
            if model_key in actor_critic_configs:
                config = actor_critic_configs[model_key].copy()
                config['class'] = model_class
                self.actor_critic_models[model_key] = config
    
    def get_value_based_models(self) -> Dict[str, Dict]:
        """Get all available value-based models"""
        return self.value_based_models
    
    def get_policy_based_models(self) -> Dict[str, Dict]:
        """Get all available policy-based models"""
        return self.policy_based_models
    
    def get_actor_critic_models(self) -> Dict[str, Dict]:
        """Get all available actor-critic models"""
        return self.actor_critic_models
    
    def get_all_models(self) -> Dict[str, Dict]:
        """Get all reinforcement learning models"""
        return {
            'value_based': self.value_based_models,
            'policy_based': self.policy_based_models,
            'actor_critic': self.actor_critic_models
        }
    
    def get_model_by_name(self, model_name: str, category: str = None):
        """Get a specific model by name"""
        if category == 'value_based':
            return self.value_based_models.get(model_name)
        elif category == 'policy_based':
            return self.policy_based_models.get(model_name)
        elif category == 'actor_critic':
            return self.actor_critic_models.get(model_name)
        else:
            # Search in all categories
            for models in [self.value_based_models, self.policy_based_models, self.actor_critic_models]:
                if model_name in models:
                    return models[model_name]
            return None
    
    def get_models_by_environment_type(self, env_type: str) -> Dict[str, Dict]:
        """Get models suitable for a specific environment type"""
        suitable_models = {}
        
        all_models = self.get_all_models()
        for category, models in all_models.items():
            for name, info in models.items():
                if env_type in info.get('environment_types', []):
                    suitable_models[f"{category}_{name}"] = info
        
        return suitable_models
    
    def get_models_by_task_type(self, task_type: str) -> Dict[str, Dict]:
        """Get models suitable for a specific task type"""
        suitable_models = {}
        
        all_models = self.get_all_models()
        for category, models in all_models.items():
            for name, info in models.items():
                if task_type in info.get('task_types', []):
                    suitable_models[f"{category}_{name}"] = info
        
        return suitable_models


class ReinforcementLearningTrainer:
    """Trainer class for reinforcement learning models"""
    
    def __init__(self):
        self.registry = ReinforcementLearningRegistry()
        self.trained_models = {}
        self.training_history = []
    
    def train_model(self, model_name: str, env, category: str, hyperparameters: Dict = None,
                   episodes: int = 100, max_steps: int = 200, verbose: bool = False) -> Dict:
        """Train a single reinforcement learning model"""
        
        model_info = self.registry.get_model_by_name(model_name, category)
        if model_info is None:
            raise ValueError(f"Model '{model_name}' not found for category '{category}'")
        
        # Initialize model with hyperparameters
        if hyperparameters is None:
            hyperparameters = {}
        
        try:
            # Get environment properties
            if hasattr(env, 'observation_space'):
                if hasattr(env.observation_space, 'n'):
                    state_size = env.observation_space.n
                else:
                    state_size = env.observation_space.shape[0]
            else:
                state_size = env.n_states if hasattr(env, 'n_states') else 10
            
            if hasattr(env, 'action_space'):
                if hasattr(env.action_space, 'n'):
                    action_size = env.action_space.n
                else:
                    action_size = env.action_space.shape[0]
            else:
                action_size = env.n_actions if hasattr(env, 'n_actions') else 4
            
            # Add state and action sizes to hyperparameters
            if category in ['value_based'] and model_name in ['qlearning', 'sarsa', 'montecarlo']:
                hyperparameters['n_states'] = state_size
                hyperparameters['n_actions'] = action_size
            else:
                hyperparameters['state_size'] = state_size
                hyperparameters['action_size'] = action_size
            
            if verbose:
                print(f"Training {model_info['name']}...")
                print(f"State size: {state_size}, Action size: {action_size}")
                print(f"Episodes: {episodes}, Max steps: {max_steps}")
            
            # Initialize model
            model = model_info['class'](**hyperparameters)
            
            # Train the model
            episode_rewards = []
            for episode in range(episodes):
                state = env.reset()
                if isinstance(state, tuple):
                    state = state[0]
                total_reward = 0
                
                # Special handling for SARSA (on-policy)
                if model_name == 'sarsa':
                    action = model.choose_action(state)
                
                for step in range(max_steps):
                    if model_name != 'sarsa':
                        action = model.choose_action(state)
                    
                    result = env.step(action)
                    
                    if len(result) == 5:
                        next_state, reward, terminated, truncated, _ = result
                        done = terminated or truncated
                    else:
                        next_state, reward, done, _ = result
                    
                    if isinstance(next_state, tuple):
                        next_state = next_state[0]
                    
                    # SARSA needs next_action
                    if model_name == 'sarsa':
                        next_action = model.choose_action(next_state)
                        model.update(state, action, reward, next_state, next_action, done)
                        action = next_action
                    elif hasattr(model, 'update'):
                        model.update(state, action, reward, next_state, done)
                    elif hasattr(model, 'remember'):
                        model.remember(state, action, reward, next_state, done)
                    elif hasattr(model, 'store_transition'):
                        # Policy-based and actor-critic methods only need state, action, reward
                        if model_name in ['policygradient', 'reinforce', 'actorcritic']:
                            model.store_transition(state, action, reward)
                        else:
                            model.store_transition(state, action, reward, done)
                    
                    state = next_state
                    total_reward += reward
                    
                    if done:
                        break
                
                episode_rewards.append(total_reward)
                
                if hasattr(model, 'replay'):
                    model.replay()
                elif hasattr(model, 'learn'):
                    # Policy-based and actor-critic methods don't take arguments
                    if model_name in ['policygradient', 'reinforce', 'actorcritic']:
                        model.learn()
                    else:
                        model.learn(next_state if not done else state)
            
            # Calculate metrics
            avg_reward = np.mean(episode_rewards[-min(100, len(episode_rewards)):])
            max_reward = np.max(episode_rewards)
            min_reward = np.min(episode_rewards)
            final_epsilon = model.epsilon if hasattr(model, 'epsilon') else 0
            
            result = {
                'model_name': model_name,
                'model_display_name': model_info['name'],
                'model_type': model_info['type'],
                'category': category,
                'model_instance': model,
                'hyperparameters': hyperparameters,
                'training_successful': True,
                'error': None,
                'episodes': episodes,
                'max_steps': max_steps,
                'avg_reward': avg_reward,
                'max_reward': max_reward,
                'min_reward': min_reward,
                'final_epsilon': final_epsilon,
                'episode_rewards': episode_rewards,
                'environment_types': model_info.get('environment_types', []),
                'task_types': model_info.get('task_types', [])
            }
            
            # Store trained model
            self.trained_models[f"{model_name}_{len(self.trained_models)}"] = result
            self.training_history.append(result)
            
            return result
            
        except Exception as e:
            import traceback
            error_result = {
                'model_name': model_name,
                'model_display_name': model_info['name'],
                'model_type': model_info['type'],
                'category': category,
                'model_instance': None,
                'hyperparameters': hyperparameters,
                'training_successful': False,
                'error': str(e),
                'error_trace': traceback.format_exc(),
                'episodes': episodes
            }
            self.training_history.append(error_result)
            return error_result
    
    def train_multiple_models(self, model_names: List[str], env, category: str,
                            episodes: int = 100, max_steps: int = 200,
                            verbose: bool = False) -> List[Dict]:
        """Train multiple models"""
        results = []
        
        for model_name in model_names:
            result = self.train_model(
                model_name, env, category, episodes=episodes,
                max_steps=max_steps, verbose=verbose
            )
            results.append(result)
        
        return results
    
    def get_training_summary(self) -> pd.DataFrame:
        """Get summary of all training results"""
        if not self.training_history:
            return pd.DataFrame()
        
        summary_data = []
        for result in self.training_history:
            summary_row = {
                'Model': result['model_display_name'],
                'Type': result['model_type'],
                'Category': result.get('category', 'Unknown'),
                'Training_Successful': result['training_successful'],
                'Episodes': result.get('episodes', 'N/A'),
                'Avg_Reward': result.get('avg_reward', 'N/A'),
                'Max_Reward': result.get('max_reward', 'N/A'),
                'Final_Epsilon': result.get('final_epsilon', 'N/A'),
                'Environment_Types': ', '.join(result.get('environment_types', [])),
                'Task_Types': ', '.join(result.get('task_types', [])),
                'Error': result.get('error', None)
            }
            summary_data.append(summary_row)
        
        return pd.DataFrame(summary_data)
    
    def get_best_model(self, metric: str = 'avg_reward') -> Optional[Dict]:
        """Get the best performing model based on a metric"""
        successful_models = [r for r in self.training_history if r['training_successful']]
        
        if not successful_models:
            return None
        
        # Filter models that have the requested metric
        models_with_metric = [r for r in successful_models if metric in r and r[metric] is not None]
        
        if not models_with_metric:
            return None
        
        # For RL, higher reward is better
        return max(models_with_metric, key=lambda x: x[metric])
    
    def clear_history(self):
        """Clear training history and trained models"""
        self.trained_models.clear()
        self.training_history.clear()


# Utility functions
def get_reinforcement_learning_info() -> Dict:
    """Get information about all available reinforcement learning models"""
    registry = ReinforcementLearningRegistry()
    return registry.get_all_models()


def create_model_comparison_report(training_results: List[Dict]) -> pd.DataFrame:
    """Create a comparison report from training results"""
    if not training_results:
        return pd.DataFrame()
    
    comparison_data = []
    for result in training_results:
        if result['training_successful']:
            row = {
                'Model': result['model_display_name'],
                'Type': result['model_type'],
                'Category': result.get('category', 'Unknown'),
                'Episodes': result.get('episodes', 'N/A'),
                'Avg_Reward': result.get('avg_reward', 'N/A'),
                'Max_Reward': result.get('max_reward', 'N/A'),
                'Final_Epsilon': result.get('final_epsilon', 'N/A'),
                'Environment_Types': ', '.join(result.get('environment_types', [])),
                'Task_Types': ', '.join(result.get('task_types', [])),
                'Hyperparameters': str(result.get('hyperparameters', {}))
            }
            comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    registry = ReinforcementLearningRegistry()
    
    print("Available Value-Based Models:")
    for name, info in registry.get_value_based_models().items():
        print(f"- {info['name']} ({info['type']}): {info['description']}")
    
    print("\nAvailable Policy-Based Models:")
    for name, info in registry.get_policy_based_models().items():
        print(f"- {info['name']} ({info['type']}): {info['description']}")
    
    print("\nAvailable Actor-Critic Models:")
    for name, info in registry.get_actor_critic_models().items():
        print(f"- {info['name']} ({info['type']}): {info['description']}")
    
    all_models = registry.get_all_models()
    total_models = sum(len(models) for models in all_models.values())
    print(f"\nTotal Models Available: {total_models}")
