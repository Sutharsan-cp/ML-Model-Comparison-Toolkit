"""
Reinforcement Learning Models Module
Consolidates all reinforcement learning algorithms for the ML Comparison Toolkit
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

# Import dictionaries
value_based_imports = {}
policy_based_imports = {}
actor_critic_imports = {}
advanced_imports = {}

# Value-based methods
try:
    from q_learning import QLearning
    value_based_imports['q_learning'] = QLearning
except ImportError as e:
    print(f"Warning: Could not import QLearning: {e}")

try:
    from sarsa import SARSA
    value_based_imports['sarsa'] = SARSA
except ImportError as e:
    print(f"Warning: Could not import SARSA: {e}")

try:
    from monte_carlo import MonteCarlo
    value_based_imports['monte_carlo'] = MonteCarlo
except ImportError as e:
    print(f"Warning: Could not import MonteCarlo: {e}")

try:
    from deep_q_network import DQN
    value_based_imports['dqn'] = DQN
except ImportError as e:
    print(f"Warning: Could not import DQN: {e}")

try:
    from double_dqn import DoubleDQN
    value_based_imports['double_dqn'] = DoubleDQN
except ImportError as e:
    print(f"Warning: Could not import DoubleDQN: {e}")

try:
    from dueling_dqn import DuelingDQN
    value_based_imports['dueling_dqn'] = DuelingDQN
except ImportError as e:
    print(f"Warning: Could not import DuelingDQN: {e}")

try:
    from prioritized_replay import PrioritizedDQN
    value_based_imports['prioritized_dqn'] = PrioritizedDQN
except ImportError as e:
    print(f"Warning: Could not import PrioritizedDQN: {e}")

# Policy-based methods
try:
    from reinforce import REINFORCE
    policy_based_imports['reinforce'] = REINFORCE
except ImportError as e:
    print(f"Warning: Could not import REINFORCE: {e}")

try:
    from policy_gradient import PolicyGradient
    policy_based_imports['policy_gradient'] = PolicyGradient
except ImportError as e:
    print(f"Warning: Could not import PolicyGradient: {e}")

try:
    from proximal_policy_optimization import PPO
    policy_based_imports['ppo'] = PPO
except ImportError as e:
    print(f"Warning: Could not import PPO: {e}")

# Actor-critic methods
try:
    from actor_critic import ActorCritic
    actor_critic_imports['actor_critic'] = ActorCritic
except ImportError as e:
    print(f"Warning: Could not import ActorCritic: {e}")

try:
    from advantage_actor_critic import A2C
    actor_critic_imports['a2c'] = A2C
except ImportError as e:
    print(f"Warning: Could not import A2C: {e}")

try:
    from asynchronous_advantage_actor_critic import A3C
    actor_critic_imports['a3c'] = A3C
except ImportError as e:
    print(f"Warning: Could not import A3C: {e}")

try:
    from deep_deterministic_policy_gradient import DDPG
    actor_critic_imports['ddpg'] = DDPG
except ImportError as e:
    print(f"Warning: Could not import DDPG: {e}")

try:
    from twin_delayed_ddpg import TwinDelayedDDPG
    actor_critic_imports['td3'] = TwinDelayedDDPG
except ImportError as e:
    print(f"Warning: Could not import TwinDelayedDDPG: {e}")

try:
    from soft_actor_critic import SoftActorCritic
    actor_critic_imports['sac'] = SoftActorCritic
except ImportError as e:
    print(f"Warning: Could not import SoftActorCritic: {e}")

# Advanced methods
try:
    from hierarchical_rl import HierarchicalRL
    advanced_imports['hierarchical_rl'] = HierarchicalRL
except ImportError as e:
    print(f"Warning: Could not import HierarchicalRL: {e}")

try:
    from meta_rl import MetaRL
    advanced_imports['meta_rl'] = MetaRL
except ImportError as e:
    print(f"Warning: Could not import MetaRL: {e}")

try:
    from inverse_rl import InverseRL
    advanced_imports['inverse_rl'] = InverseRL
except ImportError as e:
    print(f"Warning: Could not import InverseRL: {e}")

class ReinforcementLearningRegistry:
    """Registry for all reinforcement learning models"""
    
    def __init__(self):
        # Build value-based models dictionary
        self.value_based_models = {}
        
        # Define value-based model configurations
        value_based_configs = {
            'q_learning': {
                'name': 'Q-Learning',
                'type': 'value_based',
                'description': 'Tabular Q-learning algorithm',
                'environment_types': ['discrete'],
                'hyperparameters': {
                    'learningRate': [0.01, 0.1, 0.3, 0.5],
                    'discountFactor': [0.9, 0.95, 0.99],
                    'epsilon': [0.01, 0.1, 0.3],
                    'epsilonDecay': [0.99, 0.995, 0.999],
                    'minEpsilon': [0.001, 0.01, 0.05]
                }
            },
            'sarsa': {
                'name': 'SARSA',
                'type': 'value_based',
                'description': 'State-Action-Reward-State-Action algorithm',
                'environment_types': ['discrete'],
                'hyperparameters': {
                    'learningRate': [0.01, 0.1, 0.3, 0.5],
                    'discountFactor': [0.9, 0.95, 0.99],
                    'epsilon': [0.01, 0.1, 0.3],
                    'epsilonDecay': [0.99, 0.995, 0.999]
                }
            },
            'monte_carlo': {
                'name': 'Monte Carlo',
                'type': 'value_based',
                'description': 'Monte Carlo methods for RL',
                'environment_types': ['discrete'],
                'hyperparameters': {
                    'gamma': [0.9, 0.95, 0.99],
                    'epsilon': [0.01, 0.1, 0.3],
                    'epsilonDecay': [0.99, 0.995, 0.999]
                }
            },
            'dqn': {
                'name': 'Deep Q-Network',
                'type': 'value_based',
                'description': 'Deep neural network for Q-learning',
                'environment_types': ['discrete', 'continuous_state'],
                'hyperparameters': {
                    'learningRate': [0.0001, 0.001, 0.01],
                    'gamma': [0.9, 0.95, 0.99],
                    'epsilon': [0.01, 0.1, 0.3],
                    'batchSize': [16, 32, 64, 128]
                }
            },
            'double_dqn': {
                'name': 'Double DQN',
                'type': 'value_based',
                'description': 'Double Deep Q-Network to reduce overestimation',
                'environment_types': ['discrete', 'continuous_state'],
                'hyperparameters': {
                    'learningRate': [0.0001, 0.001, 0.01],
                    'gamma': [0.9, 0.95, 0.99],
                    'epsilon': [0.01, 0.1, 0.3],
                    'batchSize': [16, 32, 64]
                }
            },
            'dueling_dqn': {
                'name': 'Dueling DQN',
                'type': 'value_based',
                'description': 'DQN with separate value and advantage streams',
                'environment_types': ['discrete', 'continuous_state'],
                'hyperparameters': {
                    'learningRate': [0.0001, 0.001, 0.01],
                    'gamma': [0.9, 0.95, 0.99],
                    'epsilon': [0.01, 0.1, 0.3],
                    'batchSize': [16, 32, 64]
                }
            },
            'prioritized_dqn': {
                'name': 'Prioritized DQN',
                'type': 'value_based',
                'description': 'DQN with prioritized experience replay',
                'environment_types': ['discrete', 'continuous_state'],
                'hyperparameters': {
                    'learningRate': [0.0001, 0.001, 0.01],
                    'gamma': [0.9, 0.95, 0.99],
                    'alpha': [0.4, 0.6, 0.8],
                    'beta': [0.4, 0.6, 0.8],
                    'batchSize': [16, 32, 64]
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
            'reinforce': {
                'name': 'REINFORCE',
                'type': 'policy_based',
                'description': 'Monte Carlo policy gradient method',
                'environment_types': ['discrete', 'continuous'],
                'hyperparameters': {
                    'learningRate': [0.0001, 0.001, 0.01],
                    'gamma': [0.9, 0.95, 0.99],
                    'hiddenDim': [32, 64, 128, 256]
                }
            },
            'policy_gradient': {
                'name': 'Policy Gradient',
                'type': 'policy_based',
                'description': 'Basic policy gradient method',
                'environment_types': ['discrete', 'continuous'],
                'hyperparameters': {
                    'learningRate': [0.0001, 0.001, 0.01],
                    'gamma': [0.9, 0.95, 0.99],
                    'hiddenDim': [32, 64, 128, 256]
                }
            },
            'ppo': {
                'name': 'Proximal Policy Optimization',
                'type': 'policy_based',
                'description': 'PPO algorithm with clipped objective',
                'environment_types': ['discrete', 'continuous'],
                'hyperparameters': {
                    'learningRate': [0.0001, 0.0003, 0.001],
                    'gamma': [0.9, 0.95, 0.99],
                    'clipRatio': [0.1, 0.2, 0.3],
                    'valueCoeff': [0.1, 0.5, 1.0],
                    'entropyCoeff': [0.001, 0.01, 0.1]
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
            'actor_critic': {
                'name': 'Actor-Critic',
                'type': 'actor_critic',
                'description': 'Basic actor-critic method',
                'environment_types': ['discrete', 'continuous'],
                'hyperparameters': {
                    'learningRate': [0.0001, 0.001, 0.01],
                    'gamma': [0.9, 0.95, 0.99],
                    'hiddenDim': [32, 64, 128, 256]
                }
            },
            'a2c': {
                'name': 'Advantage Actor-Critic',
                'type': 'actor_critic',
                'description': 'A2C algorithm with advantage estimation',
                'environment_types': ['discrete', 'continuous'],
                'hyperparameters': {
                    'learningRate': [0.0001, 0.001, 0.01],
                    'gamma': [0.9, 0.95, 0.99],
                    'valueCoeff': [0.1, 0.5, 1.0],
                    'entropyCoeff': [0.001, 0.01, 0.1],
                    'hiddenDim': [32, 64, 128]
                }
            },
            'a3c': {
                'name': 'Asynchronous Advantage Actor-Critic',
                'type': 'actor_critic',
                'description': 'A3C algorithm with asynchronous updates',
                'environment_types': ['discrete', 'continuous'],
                'hyperparameters': {
                    'learningRate': [0.0001, 0.001, 0.01],
                    'gamma': [0.9, 0.95, 0.99],
                    'valueCoeff': [0.1, 0.5, 1.0],
                    'entropyCoeff': [0.001, 0.01, 0.1]
                }
            },
            'ddpg': {
                'name': 'Deep Deterministic Policy Gradient',
                'type': 'actor_critic',
                'description': 'DDPG for continuous action spaces',
                'environment_types': ['continuous'],
                'hyperparameters': {
                    'learningRate': [0.0001, 0.001, 0.01],
                    'gamma': [0.9, 0.95, 0.99],
                    'tau': [0.001, 0.005, 0.01],
                    'batchSize': [32, 64, 128]
                }
            },
            'td3': {
                'name': 'Twin Delayed DDPG',
                'type': 'actor_critic',
                'description': 'TD3 algorithm with delayed policy updates',
                'environment_types': ['continuous'],
                'hyperparameters': {
                    'learningRate': [0.0001, 0.001, 0.01],
                    'gamma': [0.9, 0.95, 0.99],
                    'tau': [0.001, 0.005, 0.01],
                    'policyDelay': [1, 2, 3],
                    'batchSize': [32, 64, 128]
                }
            },
            'sac': {
                'name': 'Soft Actor-Critic',
                'type': 'actor_critic',
                'description': 'SAC algorithm with entropy regularization',
                'environment_types': ['continuous'],
                'hyperparameters': {
                    'learningRate': [0.0001, 0.0003, 0.001],
                    'gamma': [0.9, 0.95, 0.99],
                    'tau': [0.001, 0.005, 0.01],
                    'alpha': [0.1, 0.2, 0.3],
                    'batchSize': [32, 64, 128]
                }
            }
        }
        
        # Add successfully imported actor-critic models
        for model_key, model_class in actor_critic_imports.items():
            if model_key in actor_critic_configs:
                config = actor_critic_configs[model_key].copy()
                config['class'] = model_class
                self.actor_critic_models[model_key] = config
        
        # Build advanced models dictionary
        self.advanced_models = {}
        
        # Define advanced model configurations
        advanced_configs = {
            'hierarchical_rl': {
                'name': 'Hierarchical RL',
                'type': 'advanced',
                'description': 'Hierarchical reinforcement learning',
                'environment_types': ['discrete', 'continuous'],
                'hyperparameters': {
                    'learningRate': [0.0001, 0.001, 0.01],
                    'gamma': [0.9, 0.95, 0.99],
                    'nSubgoals': [2, 4, 8],
                    'hiddenDim': [32, 64, 128]
                }
            },
            'meta_rl': {
                'name': 'Meta RL',
                'type': 'advanced',
                'description': 'Meta-learning for reinforcement learning',
                'environment_types': ['discrete', 'continuous'],
                'hyperparameters': {
                    'metaLearningRate': [0.0001, 0.001, 0.01],
                    'taskLearningRate': [0.001, 0.01, 0.1],
                    'gamma': [0.9, 0.95, 0.99],
                    'hiddenDim': [32, 64, 128]
                }
            },
            'inverse_rl': {
                'name': 'Inverse RL',
                'type': 'advanced',
                'description': 'Inverse reinforcement learning',
                'environment_types': ['discrete', 'continuous'],
                'hyperparameters': {
                    'learningRate': [0.0001, 0.001, 0.01],
                    'gamma': [0.9, 0.95, 0.99],
                    'hiddenDim': [32, 64, 128]
                }
            }
        }
        
        # Add successfully imported advanced models
        for model_key, model_class in advanced_imports.items():
            if model_key in advanced_configs:
                config = advanced_configs[model_key].copy()
                config['class'] = model_class
                self.advanced_models[model_key] = config
    
    def get_value_based_models(self) -> Dict[str, Dict]:
        """Get all available value-based models"""
        return self.value_based_models
    
    def get_policy_based_models(self) -> Dict[str, Dict]:
        """Get all available policy-based models"""
        return self.policy_based_models
    
    def get_actor_critic_models(self) -> Dict[str, Dict]:
        """Get all available actor-critic models"""
        return self.actor_critic_models
    
    def get_advanced_models(self) -> Dict[str, Dict]:
        """Get all available advanced models"""
        return self.advanced_models
    
    def get_all_models(self) -> Dict[str, Dict]:
        """Get all RL models"""
        return {
            'value_based': self.value_based_models,
            'policy_based': self.policy_based_models,
            'actor_critic': self.actor_critic_models,
            'advanced': self.advanced_models
        }
    
    def get_model_by_name(self, model_name: str, category: str = None):
        """Get a specific model by name"""
        if category == 'value_based':
            return self.value_based_models.get(model_name)
        elif category == 'policy_based':
            return self.policy_based_models.get(model_name)
        elif category == 'actor_critic':
            return self.actor_critic_models.get(model_name)
        elif category == 'advanced':
            return self.advanced_models.get(model_name)
        else:
            # Search in all categories
            for models in [self.value_based_models, self.policy_based_models, 
                          self.actor_critic_models, self.advanced_models]:
                if model_name in models:
                    return models[model_name]
            return None

class ReinforcementLearningTrainer:
    """Trainer class for reinforcement learning models"""
    
    def __init__(self):
        self.registry = ReinforcementLearningRegistry()
        self.trained_models = {}
        self.training_history = []
    
    def train_model(self, model_name: str, environment, category: str = None, 
                   hyperparameters: Dict = None, episodes: int = 100, 
                   verbose: bool = False) -> Dict:
        """Train a single RL model"""
        
        model_info = self.registry.get_model_by_name(model_name, category)
        if model_info is None:
            raise ValueError(f"Model '{model_name}' not found")
        
        # Initialize model with hyperparameters
        if hyperparameters is None:
            hyperparameters = {}
        
        try:
            # Get environment dimensions
            if hasattr(environment, 'observation_space'):
                state_dim = environment.observation_space.shape[0] if hasattr(environment.observation_space, 'shape') else environment.observation_space.n
            else:
                state_dim = getattr(environment, 'state_dim', 4)  # Default
            
            if hasattr(environment, 'action_space'):
                action_dim = environment.action_space.shape[0] if hasattr(environment.action_space, 'shape') else environment.action_space.n
            else:
                action_dim = getattr(environment, 'action_dim', 2)  # Default
            
            # Initialize model with environment dimensions
            if 'state_dim' not in hyperparameters and 'state_size' not in hyperparameters:
                if 'dqn' in model_name.lower():
                    hyperparameters['state_size'] = state_dim
                else:
                    hyperparameters['state_dim'] = state_dim
            if 'action_dim' not in hyperparameters and 'action_size' not in hyperparameters:
                if 'dqn' in model_name.lower():
                    hyperparameters['action_size'] = action_dim
                else:
                    hyperparameters['action_dim'] = action_dim
            if 'n_states' not in hyperparameters and hasattr(environment, 'observation_space'):
                if hasattr(environment.observation_space, 'n'):
                    hyperparameters['n_states'] = environment.observation_space.n
            if 'n_actions' not in hyperparameters and hasattr(environment, 'action_space'):
                if hasattr(environment.action_space, 'n'):
                    hyperparameters['n_actions'] = environment.action_space.n
            
            model = model_info['class'](**hyperparameters)
            
            if verbose:
                print(f"Training {model_info['name']} for {episodes} episodes...")
            
            # Training loop
            episode_rewards = []
            episode_lengths = []
            
            for episode in range(episodes):
                state = environment.reset()
                if isinstance(state, tuple):  # Handle new gym API
                    state = state[0]
                
                episode_reward = 0
                episode_length = 0
                done = False
                
                while not done:
                    # Get action from model
                    if hasattr(model, 'selectAction'):
                        action = model.selectAction(state)
                    elif hasattr(model, 'act'):
                        action = model.act(state)
                    else:
                        # Fallback for models without standard interface
                        action = environment.action_space.sample() if hasattr(environment, 'action_space') else 0
                    
                    # Take step in environment
                    step_result = environment.step(action)
                    if len(step_result) == 4:  # Old gym API
                        next_state, reward, done, info = step_result
                    else:  # New gym API
                        next_state, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    
                    # Update model
                    if hasattr(model, 'update'):
                        model.update(state, action, reward, next_state, done)
                    elif hasattr(model, 'learn'):
                        model.learn(state, action, reward, next_state, done)
                    
                    state = next_state
                    episode_reward += reward
                    episode_length += 1
                    
                    # Prevent infinite episodes
                    if episode_length > 1000:
                        break
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                if verbose and (episode + 1) % (episodes // 10) == 0:
                    avg_reward = np.mean(episode_rewards[-10:])
                    print(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.2f}")
            
            result = {
                'model_name': model_name,
                'model_display_name': model_info['name'],
                'model_type': model_info['type'],
                'model_category': category,
                'model_instance': model,
                'hyperparameters': hyperparameters,
                'training_successful': True,
                'error': None,
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'total_episodes': episodes,
                'final_reward': episode_rewards[-1] if episode_rewards else 0,
                'average_reward': np.mean(episode_rewards) if episode_rewards else 0,
                'best_reward': np.max(episode_rewards) if episode_rewards else 0,
                'average_length': np.mean(episode_lengths) if episode_lengths else 0
            }
            
            # Store trained model
            self.trained_models[f"{model_name}_{len(self.trained_models)}"] = result
            self.training_history.append(result)
            
            return result
            
        except Exception as e:
            error_result = {
                'model_name': model_name,
                'model_display_name': model_info['name'],
                'model_type': model_info['type'],
                'model_category': category,
                'model_instance': None,
                'hyperparameters': hyperparameters,
                'training_successful': False,
                'error': str(e)
            }
            self.training_history.append(error_result)
            return error_result
    
    def get_training_summary(self) -> pd.DataFrame:
        """Get summary of all training results"""
        if not self.training_history:
            return pd.DataFrame()
        
        summary_data = []
        for result in self.training_history:
            summary_row = {
                'Model': result['model_display_name'],
                'Type': result['model_type'],
                'Category': result.get('model_category', 'Unknown'),
                'Training_Successful': result['training_successful'],
                'Total_Episodes': result.get('total_episodes', 'N/A'),
                'Final_Reward': result.get('final_reward', 'N/A'),
                'Average_Reward': result.get('average_reward', 'N/A'),
                'Best_Reward': result.get('best_reward', 'N/A'),
                'Average_Length': result.get('average_length', 'N/A'),
                'Error': result.get('error', None)
            }
            summary_data.append(summary_row)
        
        return pd.DataFrame(summary_data)
    
    def get_best_model(self, metric: str = 'average_reward') -> Optional[Dict]:
        """Get the best performing model based on a metric"""
        successful_models = [r for r in self.training_history if r['training_successful']]
        
        if not successful_models:
            return None
        
        # Filter models that have the requested metric
        models_with_metric = [r for r in successful_models if metric in r and r[metric] != 'N/A']
        
        if not models_with_metric:
            return None
        
        return max(models_with_metric, key=lambda x: x[metric])
    
    def clear_history(self):
        """Clear training history and trained models"""
        self.trained_models.clear()
        self.training_history.clear()

# Utility functions
def get_rl_models_info() -> Dict:
    """Get information about all available RL models"""
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
                'Category': result.get('model_category', 'Unknown'),
                'Total_Episodes': result.get('total_episodes', 'N/A'),
                'Final_Reward': result.get('final_reward', 'N/A'),
                'Average_Reward': result.get('average_reward', 'N/A'),
                'Best_Reward': result.get('best_reward', 'N/A'),
                'Average_Length': result.get('average_length', 'N/A'),
                'Hyperparameters': str(result.get('hyperparameters', {}))
            }
            comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    registry = ReinforcementLearningRegistry()
    trainer = ReinforcementLearningTrainer()
    
    print("Available Value-Based Models:")
    for name, info in registry.get_value_based_models().items():
        print(f"- {info['name']} ({info['type']}): {info['description']}")
    
    print("\nAvailable Policy-Based Models:")
    for name, info in registry.get_policy_based_models().items():
        print(f"- {info['name']} ({info['type']}): {info['description']}")
    
    print("\nAvailable Actor-Critic Models:")
    for name, info in registry.get_actor_critic_models().items():
        print(f"- {info['name']} ({info['type']}): {info['description']}")
    
    print("\nAvailable Advanced Models:")
    for name, info in registry.get_advanced_models().items():
        print(f"- {info['name']} ({info['type']}): {info['description']}")