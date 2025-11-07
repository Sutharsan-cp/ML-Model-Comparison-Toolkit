"""
Streamlit Demo for Reinforcement Learning Models
Interactive dashboard for training and comparing RL algorithms
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Import our reinforcement learning module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from reinforcement_learning import ReinforcementLearningRegistry, ReinforcementLearningTrainer

# Simple Grid World Environment
class GridWorld:
    def __init__(self, size=5, goal_reward=10, step_penalty=-0.1):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4  # up, down, left, right
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
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
        reward = self.goal_reward if done else self.step_penalty
        
        return self._get_state(), reward, done, {}

def plot_learning_curve(episode_rewards, title="Learning Curve"):
    """Plot learning curve showing rewards over episodes"""
    fig = go.Figure()
    
    # Plot raw rewards
    fig.add_trace(go.Scatter(
        y=episode_rewards,
        mode='lines',
        name='Episode Reward',
        line=dict(color='lightblue', width=1),
        opacity=0.5
    ))
    
    # Plot moving average
    window = min(10, len(episode_rewards) // 10)
    if window > 1:
        moving_avg = pd.Series(episode_rewards).rolling(window=window).mean()
        fig.add_trace(go.Scatter(
            y=moving_avg,
            mode='lines',
            name=f'Moving Avg ({window} episodes)',
            line=dict(color='blue', width=2)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Episode',
        yaxis_title='Total Reward',
        hovermode='x unified'
    )
    
    return fig

def plot_performance_comparison(results):
    """Create performance comparison charts"""
    if not results:
        return None, None
    
    # Filter successful results
    successful_results = [r for r in results if r['training_successful']]
    
    if not successful_results:
        return None, None
    
    # Create comparison dataframe
    comparison_data = []
    for result in successful_results:
        comparison_data.append({
            'Model': result['model_display_name'],
            'Type': result['model_type'],
            'Avg_Reward': result.get('avg_reward', 0),
            'Max_Reward': result.get('max_reward', 0),
            'Episodes': result.get('episodes', 0)
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Bar chart for average reward
    fig1 = px.bar(df, x='Model', y='Avg_Reward', color='Type',
                  title='Average Reward Comparison',
                  labels={'Avg_Reward': 'Average Reward'})
    fig1.update_xaxes(tickangle=45)
    
    # Scatter plot for avg vs max reward
    fig2 = px.scatter(df, x='Avg_Reward', y='Max_Reward', 
                      color='Type', size='Episodes',
                      hover_data=['Model'],
                      title='Average vs Maximum Reward')
    fig2.update_xaxes(title='Average Reward')
    fig2.update_yaxes(title='Maximum Reward')
    
    return fig1, fig2

def main():
    st.set_page_config(page_title="Reinforcement Learning Demo", layout="wide")
    
    st.title("🤖 Reinforcement Learning Models Comparison")
    st.markdown("Interactive dashboard for training and comparing RL algorithms")
    
    # Show model statistics
    if 'registry' in st.session_state:
        all_models = st.session_state.registry.get_all_models()
        total_models = sum(len(models) for models in all_models.values())
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Models", total_models)
        with col2:
            st.metric("Value-Based", len(all_models.get('value_based', {})))
        with col3:
            st.metric("Policy-Based", len(all_models.get('policy_based', {})))
        with col4:
            st.metric("Actor-Critic", len(all_models.get('actor_critic', {})))
    
    # Initialize session state
    if 'training_results' not in st.session_state:
        st.session_state.training_results = []
    if 'registry' not in st.session_state:
        st.session_state.registry = ReinforcementLearningRegistry()
    if 'trainer' not in st.session_state:
        st.session_state.trainer = ReinforcementLearningTrainer()
    if 'environment' not in st.session_state:
        st.session_state.environment = None
    
    # Sidebar for configuration
    st.sidebar.header("Environment Configuration")
    
    # Environment selection
    env_type = st.sidebar.selectbox(
        "Environment Type",
        ["Grid World", "Gym Environment"],
        help="Choose the type of environment"
    )
    
    if env_type == "Grid World":
        st.sidebar.subheader("Grid World Parameters")
        grid_size = st.sidebar.slider("Grid Size", 3, 10, 5)
        goal_reward = st.sidebar.slider("Goal Reward", 1, 20, 10)
        step_penalty = st.sidebar.slider("Step Penalty", -1.0, 0.0, -0.1, 0.1)
        
        if st.sidebar.button("🎲 Create Environment"):
            st.session_state.environment = GridWorld(grid_size, goal_reward, step_penalty)
            st.sidebar.success("Grid World created!")
    
    else:  # Gym Environment
        st.sidebar.subheader("Gym Environment")
        gym_env_name = st.sidebar.selectbox(
            "Select Environment",
            ["CartPole-v1", "MountainCar-v0", "Acrobot-v1", "LunarLander-v2"],
            help="Choose a Gym environment"
        )
        
        if st.sidebar.button("🎮 Load Gym Environment"):
            try:
                import gym
                st.session_state.environment = gym.make(gym_env_name)
                st.sidebar.success(f"{gym_env_name} loaded!")
            except Exception as e:
                st.sidebar.error(f"Error loading environment: {str(e)}")
                st.sidebar.info("Make sure gym is installed: pip install gym")
    
    # Training parameters
    st.sidebar.header("Training Configuration")
    episodes = st.sidebar.slider("Number of Episodes", 10, 500, 100)
    max_steps = st.sidebar.slider("Max Steps per Episode", 50, 500, 200)
    
    # Model category selection
    category = st.sidebar.selectbox(
        "Model Category",
        ["value_based", "policy_based", "actor_critic"],
        help="Choose the category of RL models to train"
    )
    
    # Get available models for selected category
    if category == "value_based":
        available_models = st.session_state.registry.get_value_based_models()
    elif category == "policy_based":
        available_models = st.session_state.registry.get_policy_based_models()
    elif category == "actor_critic":
        available_models = st.session_state.registry.get_actor_critic_models()
    else:
        available_models = {}
    
    # Model selection
    if available_models:
        available_model_keys = list(available_models.keys())
        selected_models = st.sidebar.multiselect(
            f"Select {category.replace('_', ' ').title()} Models",
            available_model_keys,
            default=available_model_keys[:2] if len(available_model_keys) >= 2 else available_model_keys,
            help=f"Choose which {category} models to train and compare"
        )
    else:
        st.sidebar.warning(f"No {category} models available")
        selected_models = []    

    # Main content area
    if st.session_state.environment is None:
        st.info("👈 Please create or load an environment first using the sidebar controls")
        
        # Show model information
        st.subheader("📚 Available Reinforcement Learning Models")
        
        tab1, tab2, tab3 = st.tabs(["Value-Based", "Policy-Based", "Actor-Critic"])
        
        with tab1:
            value_based_models = st.session_state.registry.get_value_based_models()
            for model_key, model_info in value_based_models.items():
                with st.expander(f"🎯 {model_info['name']}"):
                    st.write(f"**Type:** {model_info['type']}")
                    st.write(f"**Description:** {model_info['description']}")
                    st.write(f"**Environment Types:** {', '.join(model_info['environment_types'])}")
                    st.write(f"**Task Types:** {', '.join(model_info['task_types'])}")
        
        with tab2:
            policy_based_models = st.session_state.registry.get_policy_based_models()
            for model_key, model_info in policy_based_models.items():
                with st.expander(f"🎯 {model_info['name']}"):
                    st.write(f"**Type:** {model_info['type']}")
                    st.write(f"**Description:** {model_info['description']}")
                    st.write(f"**Environment Types:** {', '.join(model_info['environment_types'])}")
                    st.write(f"**Task Types:** {', '.join(model_info['task_types'])}")
        
        with tab3:
            actor_critic_models = st.session_state.registry.get_actor_critic_models()
            for model_key, model_info in actor_critic_models.items():
                with st.expander(f"🎯 {model_info['name']}"):
                    st.write(f"**Type:** {model_info['type']}")
                    st.write(f"**Description:** {model_info['description']}")
                    st.write(f"**Environment Types:** {', '.join(model_info['environment_types'])}")
                    st.write(f"**Task Types:** {', '.join(model_info['task_types'])}")
        
        return
    
    # Display environment information
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🌍 Environment Overview")
        
        env = st.session_state.environment
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            if hasattr(env, 'n_states'):
                st.metric("States", env.n_states)
            elif hasattr(env, 'observation_space'):
                if hasattr(env.observation_space, 'n'):
                    st.metric("States", env.observation_space.n)
                else:
                    st.metric("State Dim", env.observation_space.shape[0])
        
        with col_b:
            if hasattr(env, 'n_actions'):
                st.metric("Actions", env.n_actions)
            elif hasattr(env, 'action_space'):
                if hasattr(env.action_space, 'n'):
                    st.metric("Actions", env.action_space.n)
                else:
                    st.metric("Action Dim", env.action_space.shape[0])
        
        with col_c:
            st.metric("Episodes", episodes)
        
        # Environment info
        if isinstance(env, GridWorld):
            st.info(f"🎮 **Grid World Environment:** {env.size}x{env.size} grid, Goal at ({env.goal_pos[0]}, {env.goal_pos[1]})")
        else:
            st.info(f"🎮 **Gym Environment:** {env.spec.id if hasattr(env, 'spec') else 'Custom'}")
    
    with col2:
        st.subheader("📋 Selected Models")
        if selected_models:
            for model_name in selected_models:
                model_info = available_models[model_name]
                with st.expander(f"{model_info['name']}", expanded=False):
                    st.write(f"**Type:** {model_info['type']}")
                    st.write(f"**Description:** {model_info['description'][:80]}...")
        else:
            st.info("No models selected. Please select models from the sidebar.")
    
    # Training section
    st.subheader("🚀 Model Training")
    
    col_train, col_clear = st.columns([3, 1])
    
    with col_train:
        train_button = st.button("🎯 Train Selected Models", disabled=not selected_models, use_container_width=True)
    
    with col_clear:
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.training_results = []
            st.session_state.trainer.clear_history()
            st.success("Results cleared!")
            st.rerun()
    
    if train_button and selected_models:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        new_results = []
        
        for i, model_name in enumerate(selected_models):
            status_text.text(f"Training {model_name}... ({i+1}/{len(selected_models)})")
            progress_bar.progress((i) / len(selected_models))
            
            try:
                # Set appropriate hyperparameters based on model
                hyperparams = {}
                
                if model_name in ['qlearning', 'sarsa']:
                    hyperparams = {
                        'learning_rate': 0.1,
                        'discount_factor': 0.99,
                        'exploration_rate': 1.0,
                        'exploration_decay': 0.995,
                        'min_exploration': 0.01
                    }
                elif model_name == 'montecarlo':
                    hyperparams = {
                        'discount_factor': 0.99,
                        'exploration_rate': 0.2
                    }
                elif model_name in ['dqn', 'doubledqn']:
                    hyperparams = {
                        'learning_rate': 0.001,
                        'discount_factor': 0.99,
                        'exploration_rate': 1.0,
                        'exploration_decay': 0.995,
                        'batch_size': 32,
                        'target_update_freq': 10
                    }
                elif model_name in ['policygradient', 'reinforce']:
                    hyperparams = {
                        'learning_rate': 0.001,
                        'discount_factor': 0.99
                    }
                elif model_name == 'ppo':
                    hyperparams = {
                        'actor_lr': 0.0003,
                        'critic_lr': 0.001,
                        'discount_factor': 0.99,
                        'gae_lambda': 0.95,
                        'clip_ratio': 0.2,
                        'epochs': 10,
                        'batch_size': 64
                    }
                elif model_name in ['actorcritic', 'ddpg', 'sac']:
                    hyperparams = {
                        'actor_lr': 0.0003,
                        'critic_lr': 0.001,
                        'discount_factor': 0.99
                    }
                
                result = st.session_state.trainer.train_model(
                    model_name,
                    env,
                    category,
                    hyperparams,
                    episodes=episodes,
                    max_steps=max_steps,
                    verbose=False
                )
                new_results.append(result)
                
            except Exception as e:
                st.error(f"Error training {model_name}: {str(e)}")
                # Add failed result
                new_results.append({
                    'model_name': model_name,
                    'model_display_name': available_models[model_name]['name'],
                    'training_successful': False,
                    'error': str(e)
                })
        
        progress_bar.progress(1.0)
        status_text.text("✅ Training completed!")
        
        # Add new results to session state
        st.session_state.training_results.extend(new_results)
        
        successful_count = len([r for r in new_results if r['training_successful']])
        st.success(f"Successfully trained {successful_count} out of {len(new_results)} models!")
        
        # Auto-rerun to show results
        st.rerun()    
  
  # Display results
    if st.session_state.training_results:
        st.subheader("📈 Training Results")
        
        # Results summary table
        summary_data = []
        for result in st.session_state.training_results:
            row = {
                'Model': result['model_display_name'],
                'Type': result.get('model_type', 'N/A'),
                'Status': '✅ Success' if result['training_successful'] else '❌ Failed'
            }
            
            if result['training_successful']:
                row.update({
                    'Avg Reward': f"{result.get('avg_reward', 0):.2f}",
                    'Max Reward': f"{result.get('max_reward', 0):.2f}",
                    'Min Reward': f"{result.get('min_reward', 0):.2f}",
                    'Final Epsilon': f"{result.get('final_epsilon', 0):.3f}",
                    'Episodes': result.get('episodes', 'N/A')
                })
            else:
                row['Error'] = result.get('error', 'Unknown error')[:50] + '...' if len(result.get('error', '')) > 50 else result.get('error', '')
            
            summary_data.append(row)
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True)
        
        # Performance comparison
        st.subheader("🏆 Performance Comparison")
        
        fig1, fig2 = plot_performance_comparison(st.session_state.training_results)
        
        if fig1 and fig2:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                st.plotly_chart(fig2, use_container_width=True)
        
        # Best model
        best_model = st.session_state.trainer.get_best_model('avg_reward')
        if best_model:
            st.subheader("🥇 Best Performing Model")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Model", best_model['model_display_name'])
            with col2:
                st.metric("Avg Reward", f"{best_model['avg_reward']:.2f}")
            with col3:
                st.metric("Max Reward", f"{best_model['max_reward']:.2f}")
            with col4:
                st.metric("Episodes", best_model['episodes'])
            
            # Learning curve for best model
            if 'episode_rewards' in best_model:
                st.subheader("📊 Best Model Learning Curve")
                fig_learning = plot_learning_curve(
                    best_model['episode_rewards'],
                    f"Learning Curve - {best_model['model_display_name']}"
                )
                st.plotly_chart(fig_learning, use_container_width=True)
        
        # Learning curves for all models
        st.subheader("📈 All Models Learning Curves")
        
        successful_results = [r for r in st.session_state.training_results 
                            if r['training_successful'] and 'episode_rewards' in r]
        
        if successful_results:
            fig_all = go.Figure()
            
            for result in successful_results:
                episode_rewards = result['episode_rewards']
                window = min(10, len(episode_rewards) // 10)
                if window > 1:
                    moving_avg = pd.Series(episode_rewards).rolling(window=window).mean()
                    fig_all.add_trace(go.Scatter(
                        y=moving_avg,
                        mode='lines',
                        name=result['model_display_name']
                    ))
            
            fig_all.update_layout(
                title='Learning Curves Comparison (Moving Average)',
                xaxis_title='Episode',
                yaxis_title='Average Reward',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_all, use_container_width=True)
        
        # Detailed comparison table
        st.subheader("📊 Detailed Model Comparison")
        
        comparison_df = st.session_state.trainer.get_training_summary()
        
        if not comparison_df.empty:
            st.dataframe(comparison_df, use_container_width=True)
            
            # Download button for results
            csv = comparison_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="reinforcement_learning_results.csv",
                mime="text/csv"
            )
    
    else:
        st.info("👆 Select models and click 'Train Selected Models' to see results")
    
    # Model information section
    with st.expander("📚 Reinforcement Learning Methods Overview", expanded=False):
        st.markdown("""
        ### Reinforcement Learning Techniques
        
        RL agents learn by interacting with an environment and receiving rewards.
        
        #### 🎯 Value-Based Methods
        - **Q-Learning**: Off-policy TD control for discrete states/actions
        - **SARSA**: On-policy TD control algorithm
        - **Monte Carlo**: Episode-based learning using complete returns
        - **DQN**: Deep Q-Network with experience replay
        - **Double DQN**: Reduces overestimation bias in Q-learning
        
        #### 🎮 Policy-Based Methods
        - **Policy Gradient**: Direct policy optimization
        - **REINFORCE**: Monte Carlo policy gradient
        - **PPO**: Proximal Policy Optimization with clipped objective
        
        #### 🤝 Actor-Critic Methods
        - **Actor-Critic**: Combines value and policy methods
        - **DDPG**: Deep Deterministic Policy Gradient for continuous actions
        - **SAC**: Soft Actor-Critic with maximum entropy
        
        ### Key Concepts
        - **Exploration vs Exploitation**: Balance between trying new actions and using known good actions
        - **Discount Factor (γ)**: How much to value future rewards
        - **Learning Rate (α)**: Step size for updates
        - **Epsilon (ε)**: Exploration rate in epsilon-greedy policies
        
        ### When to Use
        - **Q-Learning/SARSA**: Simple discrete environments
        - **DQN**: Complex environments with continuous states
        - **PPO**: General-purpose, stable training
        - **DDPG/SAC**: Continuous action spaces
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Features:**
    - 🎮 **Grid World Environment**: Simple discrete environment for testing
    - 🏋️ **Gym Integration**: Support for OpenAI Gym environments
    - 🤖 **11 RL Algorithms**: Value-based, policy-based, and actor-critic methods
    - 📈 **Learning Curves**: Visualize training progress
    - 🏆 **Model Comparison**: Side-by-side performance analysis
    - 📥 **Export Results**: Download comparison results as CSV
    
    **Supported Models:**
    - Value-Based: Q-Learning, SARSA, Monte Carlo, DQN, Double DQN
    - Policy-Based: Policy Gradient, REINFORCE, PPO
    - Actor-Critic: Actor-Critic, DDPG, SAC
    """)

if __name__ == "__main__":
    main()
