# Reinforcement Learning Module

A comprehensive reinforcement learning module for the ML Comparison Toolkit that provides a unified interface for training and comparing various RL algorithms.

## Overview

This module consolidates all reinforcement learning algorithms from the `backend/app/models/reinforcement` directory and provides:

- **Unified Registry**: Access to all RL models with their configurations
- **Automated Training**: Easy training and evaluation of multiple agents
- **Performance Comparison**: Built-in comparison and ranking of agents
- **Flexible Configuration**: Support for custom hyperparameters and environments

## Available Models

### Value-Based Methods (6 models)

1. **Q-Learning** - Tabular off-policy TD control algorithm
2. **SARSA** - On-policy TD control algorithm
3. **Monte Carlo** - Episode-based learning using complete returns
4. **Deep Q-Network (DQN)** - Deep learning-based Q-learning with experience replay
5. **Double DQN** - DQN with double Q-learning to reduce overestimation
6. **Dueling DQN** - DQN with separate value and advantage streams

### Policy-Based Methods (3 models)

1. **Policy Gradient** - Direct policy optimization using gradient ascent
2. **REINFORCE** - Monte Carlo policy gradient algorithm
3. **Proximal Policy Optimization (PPO)** - Policy optimization with clipped surrogate objective

### Actor-Critic Methods (5 models)

1. **Actor-Critic** - Combines value-based and policy-based methods
2. **A3C** - Asynchronous Advantage Actor-Critic with parallel updates
3. **DDPG** - Deep Deterministic Policy Gradient for continuous actions
4. **TD3** - Twin Delayed DDPG with twin critics
5. **SAC** - Soft Actor-Critic with maximum entropy

### Advanced Methods (2 models)

1. **Hierarchical RL** - Multi-level policy hierarchy for complex tasks
2. **Inverse RL** - Learn reward function from expert demonstrations

## Quick Start

### Basic Usage

```python
from reinforcement_learning import ReinforcementLearningRegistry, ReinforcementLearningTrainer
import gym

# Initialize
registry = ReinforcementLearningRegistry()
trainer = ReinforcementLearningTrainer()

# Create environment
env = gym.make('CartPole-v1')

# Train a single model
result = trainer.train_model(
    'dqn',
    env,
    category='value_based',
    hyperparameters={
        'learning_rate': 0.001,
        'discount_factor': 0.99,
        'batch_size': 64
    },
    episodes=500,
    max_steps=200,
    verbose=True
)

print(f"Average Reward: {result['avg_reward']:.2f}")
```

### Training Multiple Models

```python
# Train all value-based models
results = trainer.train_category_models(
    'value_based',
    env,
    episodes=500,
    max_steps=200,
    verbose=True
)

# Get training summary
summary = trainer.get_training_summary()
print(summary)

# Get best model
best_model = trainer.get_best_model('avg_reward')
print(f"Best Model: {best_model['model_display_name']}")
print(f"Average Reward: {best_model['avg_reward']:.2f}")
```

### Model Registry

```python
# Get all available models
all_models = registry.get_all_models()

# Get value-based models only
value_based_models = registry.get_value_based_models()

# Get model by name
model_info = registry.get_model_by_name('dqn', 'value_based')
print(f"Model: {model_info['name']}")
print(f"Description: {model_info['description']}")
print(f"Hyperparameters: {model_info['hyperparameters']}")

# Get models by environment type
discrete_models = registry.get_models_by_environment_type('discrete_action')

# Get models by task type
control_models = registry.get_models_by_task_type('control')
```

## Features

### 1. Unified Model Registry

The `ReinforcementLearningRegistry` class provides:
- Centralized access to all RL models
- Model metadata (name, type, description, supported environments)
- Hyperparameter configurations
- Filtering by environment type or task type

### 2. Automated Training

The `ReinforcementLearningTrainer` class provides:
- Single model training with custom hyperparameters
- Batch training of multiple models
- Training all models in a category
- Automatic metric calculation

### 3. Performance Metrics

- **Average Reward**: Mean reward over last 100 episodes
- **Final Epsilon**: Exploration rate at end of training
- **Training History**: Episode-by-episode performance
- **Convergence Analysis**: Learning curves and stability

### 4. Model Comparison

- Training summary with all metrics
- Best model selection based on any metric
- Side-by-side comparison of models
- Automatic sorting by performance

## Model Categories

### Value-Based Methods

Learn action-value functions (Q-values) to determine optimal actions.

**When to use:**
- Discrete action spaces
- Need for sample efficiency
- Offline learning from replay buffer

**Best models:**
- **Q-Learning**: Simple tabular environments
- **DQN**: Complex environments with continuous states
- **Double DQN**: When overestimation is a problem

### Policy-Based Methods

Directly optimize the policy without learning value functions.

**When to use:**
- Continuous action spaces
- Stochastic policies needed
- High-dimensional action spaces

**Best models:**
- **REINFORCE**: Simple episodic tasks
- **PPO**: General-purpose, stable training

### Actor-Critic Methods

Combine value-based and policy-based approaches.

**When to use:**
- Need for both policy and value estimates
- Continuous control tasks
- Sample-efficient learning

**Best models:**
- **Actor-Critic**: Basic continuous control
- **DDPG**: Continuous actions, deterministic policies
- **SAC**: Robust learning with exploration

### Advanced Methods

Specialized techniques for complex scenarios.

**When to use:**
- Hierarchical task structures
- Learning from demonstrations
- Multi-task learning

## Environment Types

### Discrete State, Discrete Action
- **Models**: Q-Learning, SARSA, Monte Carlo
- **Examples**: Grid worlds, simple games
- **Characteristics**: Tabular methods, exact solutions

### Continuous State, Discrete Action
- **Models**: DQN, Double DQN, Dueling DQN, PPO
- **Examples**: CartPole, Atari games
- **Characteristics**: Function approximation, deep learning

### Continuous State, Continuous Action
- **Models**: DDPG, TD3, SAC
- **Examples**: Robot control, autonomous driving
- **Characteristics**: Deterministic/stochastic policies

## Hyperparameter Tuning

### Q-Learning / SARSA
```python
hyperparameters = {
    'learning_rate': 0.1,        # Step size for updates
    'discount_factor': 0.99,     # Future reward importance
    'exploration_rate': 1.0,     # Initial exploration
    'exploration_decay': 0.995,  # Exploration decay rate
    'min_exploration': 0.01      # Minimum exploration
}
```

### DQN
```python
hyperparameters = {
    'learning_rate': 0.001,
    'discount_factor': 0.99,
    'exploration_rate': 1.0,
    'exploration_decay': 0.995,
    'batch_size': 64,
    'target_update_freq': 100
}
```

### PPO
```python
hyperparameters = {
    'actor_lr': 0.0003,
    'critic_lr': 0.001,
    'discount_factor': 0.99,
    'gae_lambda': 0.95,
    'clip_ratio': 0.2,
    'epochs': 10,
    'batch_size': 64
}
```

### DDPG / TD3 / SAC
```python
hyperparameters = {
    'actor_lr': 0.0001,
    'critic_lr': 0.001,
    'discount_factor': 0.99,
    'tau': 0.005,              # Soft update coefficient
    'batch_size': 128
}
```

## Advanced Usage

### Custom Environment

```python
class CustomEnv:
    def __init__(self):
        self.n_states = 10
        self.n_actions = 4
        self.state = 0
    
    def reset(self):
        self.state = 0
        return self.state
    
    def step(self, action):
        # Environment logic
        next_state = (self.state + action) % self.n_states
        reward = 1.0 if next_state == self.n_states - 1 else 0.0
        done = next_state == self.n_states - 1
        
        self.state = next_state
        return next_state, reward, done, {}

# Use with trainer
env = CustomEnv()
result = trainer.train_model('qlearning', env, 'value_based', episodes=1000)
```

### Gym Environment

```python
import gym

# Classic control
env = gym.make('CartPole-v1')
result = trainer.train_model('dqn', env, 'value_based', episodes=500)

# Continuous control
env = gym.make('Pendulum-v1')
result = trainer.train_model('ddpg', env, 'actor_critic', episodes=500)

# Atari games
env = gym.make('Pong-v0')
result = trainer.train_model('dqn', env, 'value_based', episodes=1000)
```

### Model Comparison Report

```python
# Train multiple models
results = trainer.train_multiple_models(
    ['qlearning', 'sarsa', 'dqn'],
    env,
    'value_based',
    episodes=500
)

# Create comparison report
from reinforcement_learning import create_model_comparison_report
report = create_model_comparison_report(results)
print(report)
```

## Performance Tips

1. **Start Simple**: Begin with Q-Learning or SARSA for discrete environments
2. **Use DQN**: For complex state spaces with discrete actions
3. **Try PPO**: General-purpose algorithm with stable training
4. **Tune Exploration**: Balance exploration vs exploitation carefully
5. **Monitor Learning**: Track average reward over episodes
6. **Adjust Learning Rate**: Lower for stability, higher for faster learning
7. **Use Target Networks**: For DQN variants, update target network periodically

## Common Issues

### Slow Convergence
- Increase learning rate
- Adjust exploration parameters
- Use experience replay (DQN)
- Try different network architectures

### Unstable Training
- Decrease learning rate
- Use target networks
- Increase batch size
- Add gradient clipping

### Poor Performance
- Tune hyperparameters
- Increase training episodes
- Check reward shaping
- Verify environment implementation

## Integration with ML Comparison Toolkit

This module integrates seamlessly with the ML Comparison Toolkit:

```python
# Use with the main comparison framework
from reinforcement_learning import get_reinforcement_learning_info

# Get all model information
model_info = get_reinforcement_learning_info()

# Use in automated model selection
trainer = ReinforcementLearningTrainer()
results = trainer.train_category_models('value_based', env, episodes=500)
best = trainer.get_best_model()
```

## Requirements

- numpy
- pandas
- tensorflow (for deep learning models)
- gym (for standard RL environments)
- Custom RL models from `backend/app/models/reinforcement`

## Testing

Run the module directly to see available models:

```bash
python backend/app/reinforcement_learning/reinforcement_learning.py
```

## License

Part of the ML Comparison Toolkit project.

## Contributing

To add new RL models:
1. Implement the model in `backend/app/models/reinforcement`
2. Add import statement in `reinforcement_learning.py`
3. Add model configuration in the registry
4. Update this README

## Support

For issues or questions, please refer to the main ML Comparison Toolkit documentation.

---

**Happy Reinforcement Learning! 🤖**