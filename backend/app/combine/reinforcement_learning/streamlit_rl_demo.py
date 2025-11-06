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

# Import our RL module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from reinforcement_learning import ReinforcementLearningRegistry, ReinforcementLearningTrainer

# Simple environments for demo
class SimpleGridWorld:
    """Simple grid world environment"""
    
    def __init__(self, size=4):
        self.size = size
        self.state_dim = size * size
        self.action_dim = 4
        self.reset()
    
    def reset(self):
        self.agent_pos = [0, 0]
        self.goal_pos = [self.size-1, self.size-1]
        self.step_count = 0
        return self._get_state()
    
    def _get_state(self):
        return self.agent_pos[0] * self.size + self.agent_pos[1]
    
    def step(self, action):
        self.step_count += 1
        old_pos = self.agent_pos.copy()
        
        if action == 0 and self.agent_pos[0] > 0:  # up
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[0] < self.size-1:  # down
            self.agent_pos[0] += 1
        elif action == 2 and self.agent_pos[1] > 0:  # left
            self.agent_pos[1] -= 1
        elif action == 3 and self.agent_pos[1] < self.size-1:  # right
            self.agent_pos[1] += 1
        
        if self.agent_pos == self.goal_pos:
            reward = 10
            done = True
        else:
            reward = -0.1
            done = self.step_count > 100  # Prevent infinite episodes
        
        return self._get_state(), reward, done, {}

class SimpleContinuousEnv:
    """Simple continuous environment"""
    
    def __init__(self):
        self.state_dim = 2
        self.action_dim = 1
        self.reset()
    
    def reset(self):
        self.state = np.random.uniform(-2, 2, 2)
        self.step_count = 0
        return self.state.copy()
    
    def step(self, action):
        self.step_count += 1
        action = np.clip(action, -1, 1)
        self.state[0] += 0.1 * action
        self.state[1] += 0.1 * (0.5 - self.state[1])
        
        reward = -np.linalg.norm(self.state)
        done = np.linalg.norm(self.state) < 0.1 or self.step_count > 100
        
        return self.state.copy(), reward, done, {}

def create_environment(env_type, env_size=4):
    """Create environment based on type"""
    if env_type == "GridWorld":
        return SimpleGridWorld(size=env_size)
    elif env_type == "Continuous":
        return SimpleContinuousEnv()
    else:
        return SimpleGridWorld(size=env_size)

def plot_training_curves(results):
    """Plot training curves for multiple models"""
    if not results:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Episode Rewards', 'Cumulative Rewards', 'Episode Lengths', 'Moving Average Rewards'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = px.colors.qualitative.Set1
    
    for i, result in enumerate(results):
        if not result['training_successful']:
            continue
            
        model_name = result['model_display_name']
        episode_rewards = result.get('episode_rewards', [])
        episode_lengths = result.get('episode_lengths', [])
        
        if not episode_rewards:
            continue
        
        episodes = list(range(1, len(episode_rewards) + 1))
        cumulative_rewards = np.cumsum(episode_rewards)
        
        # Moving average (window=10)
        window = min(10, len(episode_rewards))
        moving_avg = pd.Series(episode_rewards).rolling(window=window, min_periods=1).mean()
        
        color = colors[i % len(colors)]
        
        # Episode rewards
        fig.add_trace(
            go.Scatter(x=episodes, y=episode_rewards, name=f'{model_name} - Rewards',
                      line=dict(color=color), opacity=0.7),
            row=1, col=1
        )
        
        # Cumulative rewards
        fig.add_trace(
            go.Scatter(x=episodes, y=cumulative_rewards, name=f'{model_name} - Cumulative',
                      line=dict(color=color), showlegend=False),
            row=1, col=2
        )
        
        # Episode lengths
        fig.add_trace(
            go.Scatter(x=episodes, y=episode_lengths, name=f'{model_name} - Lengths',
                      line=dict(color=color), showlegend=False),
            row=2, col=1
        )
        
        # Moving average
        fig.add_trace(
            go.Scatter(x=episodes, y=moving_avg, name=f'{model_name} - Moving Avg',
                      line=dict(color=color, width=3), showlegend=False),
            row=2, col=2
        )
    
    fig.update_layout(height=600, title_text="Training Performance Comparison")
    fig.update_xaxes(title_text="Episode", row=2, col=1)
    fig.update_xaxes(title_text="Episode", row=2, col=2)
    fig.update_yaxes(title_text="Reward", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Reward", row=1, col=2)
    fig.update_yaxes(title_text="Episode Length", row=2, col=1)
    fig.update_yaxes(title_text="Moving Avg Reward", row=2, col=2)
    
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
            'Final_Reward': result.get('final_reward', 0),
            'Average_Reward': result.get('average_reward', 0),
            'Best_Reward': result.get('best_reward', 0),
            'Average_Length': result.get('average_length', 0)
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Bar chart for average rewards
    fig1 = px.bar(df, x='Model', y='Average_Reward', color='Type',
                  title='Average Reward Comparison',
                  labels={'Average_Reward': 'Average Reward'})
    fig1.update_xaxes(tickangle=45)
    
    # Scatter plot for reward vs length
    fig2 = px.scatter(df, x='Average_Length', y='Average_Reward', 
                      color='Type', size='Best_Reward',
                      hover_data=['Model'],
                      title='Average Reward vs Episode Length')
    
    return fig1, fig2

def main():
    st.set_page_config(page_title="RL Models Demo", layout="wide")
    
    st.title("🤖 Reinforcement Learning Models Comparison")
    st.markdown("Interactive dashboard for training and comparing RL algorithms")
    
    # Initialize session state
    if 'training_results' not in st.session_state:
        st.session_state.training_results = []
    if 'registry' not in st.session_state:
        st.session_state.registry = ReinforcementLearningRegistry()
    if 'trainer' not in st.session_state:
        st.session_state.trainer = ReinforcementLearningTrainer()
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Environment selection
    env_type = st.sidebar.selectbox(
        "Environment Type",
        ["GridWorld", "Continuous"],
        help="Choose the environment for training"
    )
    
    if env_type == "GridWorld":
        env_size = st.sidebar.slider("Grid Size", 3, 6, 4)
        env = create_environment(env_type, env_size)
        st.sidebar.info(f"GridWorld: {env_size}x{env_size} grid, goal at ({env_size-1},{env_size-1})")
    else:
        env = create_environment(env_type)
        st.sidebar.info("Continuous: 2D environment, goal at origin")
    
    # Training parameters
    episodes = st.sidebar.slider("Training Episodes", 10, 200, 50)
    
    # Model category selection
    category = st.sidebar.selectbox(
        "Model Category",
        ["value_based", "policy_based", "actor_critic", "advanced"],
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
        available_models = st.session_state.registry.get_advanced_models()
    
    # Model selection
    if available_models:
        selected_models = st.sidebar.multiselect(
            f"Select {category.replace('_', ' ').title()} Models",
            list(available_models.keys()),
            default=list(available_models.keys())[:3],
            help=f"Choose which {category} models to train and compare"
        )
    else:
        st.sidebar.warning(f"No {category} models available")
        selected_models = []
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Available Models")
        if available_models:
            for model_key, model_info in available_models.items():
                with st.expander(f"{model_info['name']}"):
                    st.write(f"**Type:** {model_info['type']}")
                    st.write(f"**Description:** {model_info['description']}")
                    st.write(f"**Environment Types:** {', '.join(model_info['environment_types'])}")
        else:
            st.info("No models available for selected category")
    
    with col1:
        st.subheader("Training & Results")
        
        # Training button
        if st.button("🚀 Train Selected Models", disabled=not selected_models):
            if selected_models:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                new_results = []
                
                for i, model_name in enumerate(selected_models):
                    status_text.text(f"Training {model_name}...")
                    progress_bar.progress((i) / len(selected_models))
                    
                    try:
                        # Set appropriate hyperparameters based on model and environment
                        hyperparams = {}
                        
                        if category == "value_based":
                            if model_name in ['q_learning', 'sarsa', 'monte_carlo']:
                                hyperparams = {
                                    'n_states': env.state_dim,
                                    'n_actions': env.action_dim,
                                    'learning_rate': 0.1,
                                    'exploration_rate': 0.1
                                }
                            elif 'dqn' in model_name:
                                hyperparams = {
                                    'state_size': env.state_dim,
                                    'action_size': env.action_dim,
                                    'learning_rate': 0.001,
                                    'exploration_rate': 0.1,
                                    'batch_size': 32
                                }
                        elif category in ["policy_based", "actor_critic"]:
                            hyperparams = {
                                'state_dim': env.state_dim,
                                'action_dim': env.action_dim,
                                'learning_rate': 0.001
                            }
                            if model_name in ['actor_critic', 'a2c', 'a3c', 'reinforce', 'policy_gradient']:
                                hyperparams['hidden_dim'] = 64
                            elif model_name in ['ddpg', 'td3', 'sac']:
                                hyperparams['batch_size'] = 32
                        elif category == "advanced":
                            hyperparams = {
                                'state_dim': env.state_dim,
                                'action_dim': env.action_dim,
                                'learning_rate': 0.001,
                                'hidden_dim': 64
                            }
                        
                        result = st.session_state.trainer.train_model(
                            model_name, env, category, hyperparams, episodes=episodes, verbose=False
                        )
                        new_results.append(result)
                        
                    except Exception as e:
                        st.error(f"Error training {model_name}: {str(e)}")
                
                progress_bar.progress(1.0)
                status_text.text("Training completed!")
                
                # Add new results to session state
                st.session_state.training_results.extend(new_results)
                
                st.success(f"Trained {len(new_results)} models successfully!")
        
        # Clear results button
        if st.button("🗑️ Clear Results"):
            st.session_state.training_results = []
            st.session_state.trainer.clear_history()
            st.success("Results cleared!")
    
    # Display results
    if st.session_state.training_results:
        st.subheader("📊 Training Results")
        
        # Results summary table
        summary_data = []
        for result in st.session_state.training_results:
            summary_data.append({
                'Model': result['model_display_name'],
                'Type': result['model_type'],
                'Status': '✅ Success' if result['training_successful'] else '❌ Failed',
                'Final Reward': f"{result.get('final_reward', 'N/A'):.2f}" if isinstance(result.get('final_reward'), (int, float)) else 'N/A',
                'Avg Reward': f"{result.get('average_reward', 'N/A'):.2f}" if isinstance(result.get('average_reward'), (int, float)) else 'N/A',
                'Best Reward': f"{result.get('best_reward', 'N/A'):.2f}" if isinstance(result.get('best_reward'), (int, float)) else 'N/A',
                'Avg Length': f"{result.get('average_length', 'N/A'):.1f}" if isinstance(result.get('average_length'), (int, float)) else 'N/A',
                'Episodes': result.get('total_episodes', 'N/A'),
                'Error': result.get('error', '')[:50] + '...' if result.get('error') and len(result.get('error', '')) > 50 else result.get('error', '')
            })
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True)
        
        # Training curves
        st.subheader("📈 Training Curves")
        training_fig = plot_training_curves(st.session_state.training_results)
        if training_fig:
            st.plotly_chart(training_fig, use_container_width=True)
        
        # Performance comparison
        st.subheader("🏆 Performance Comparison")
        col1, col2 = st.columns(2)
        
        fig1, fig2 = plot_performance_comparison(st.session_state.training_results)
        
        if fig1:
            with col1:
                st.plotly_chart(fig1, use_container_width=True)
        
        if fig2:
            with col2:
                st.plotly_chart(fig2, use_container_width=True)
        
        # Best model
        best_model = st.session_state.trainer.get_best_model('average_reward')
        if best_model:
            st.subheader("🥇 Best Performing Model")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Model", best_model['model_display_name'])
            with col2:
                st.metric("Average Reward", f"{best_model['average_reward']:.2f}")
            with col3:
                st.metric("Best Reward", f"{best_model['best_reward']:.2f}")
            with col4:
                st.metric("Avg Episode Length", f"{best_model['average_length']:.1f}")
    
    else:
        st.info("👆 Select models and click 'Train Selected Models' to see results")
    
    # Footer
    st.markdown("---")
    st.markdown("**Note:** This demo uses simplified environments for demonstration. "
                "Real RL applications may require more complex environments and longer training times.")

if __name__ == "__main__":
    main()