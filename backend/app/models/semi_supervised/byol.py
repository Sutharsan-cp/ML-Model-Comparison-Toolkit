import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class BYOL:
    def __init__(self, online_network, target_network=None, 
                 moving_average_decay=0.99, projection_dim=256, 
                 hidden_dim=4096, epochs=100, batch_size=256, 
                 learning_rate=0.001, device='cpu'):
        self.online_network = online_network
        self.target_network = target_network if target_network is not None else self._copy_network(online_network)
        self.moving_average_decay = moving_average_decay
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        
        # Add projection and prediction heads
        self.online_projector = self._build_projector(online_network.output_dim, hidden_dim, projection_dim)
        self.online_predictor = self._build_predictor(projection_dim, hidden_dim, projection_dim)
        
        self.target_projector = self._build_projector(online_network.output_dim, hidden_dim, projection_dim)
        
        # Initialize target network with online network weights
        self._update_target_network(0.0)
        
    def _copy_network(self, network):
        """Create a copy of the network"""
        return type(network)(**network.get_config() if hasattr(network, 'get_config') else {})
    
    def _build_projector(self, input_dim, hidden_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def _build_predictor(self, input_dim, hidden_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def _update_target_network(self, decay):
        """Update target network with exponential moving average"""
        for online_param, target_param in zip(self.online_network.parameters(), 
                                            self.target_network.parameters()):
            target_param.data = decay * target_param.data + (1 - decay) * online_param.data
        
        for online_param, target_param in zip(self.online_projector.parameters(),
                                            self.target_projector.parameters()):
            target_param.data = decay * target_param.data + (1 - decay) * online_param.data
    
    def _cosine_similarity_loss(self, x, y):
        """Compute cosine similarity loss"""
        x = nn.functional.normalize(x, dim=1)
        y = nn.functional.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=1).mean()
    
    def _augment(self, x):
        """Apply augmentations (simplified version)"""
        # In practice, use strong augmentations like random crop, color jitter, etc.
        aug1 = x + 0.1 * torch.randn_like(x)
        aug2 = x * (0.8 + 0.4 * torch.rand(1).to(self.device))
        return aug1, aug2
    
    def fit(self, X):
        X_tensor = torch.FloatTensor(X).to(self.device)
        n_samples = len(X_tensor)
        
        optimizer = optim.Adam(
            list(self.online_network.parameters()) + 
            list(self.online_projector.parameters()) + 
            list(self.online_predictor.parameters()), 
            lr=self.learning_rate
        )
        
        for epoch in range(self.epochs):
            self.online_network.train()
            self.target_network.train()
            self.online_projector.train()
            self.online_predictor.train()
            self.target_projector.train()
            
            total_loss = 0
            
            # Shuffle data
            indices = torch.randperm(n_samples)
            
            for i in range(0, n_samples, self.batch_size):
                batch_indices = indices[i:min(i+self.batch_size, n_samples)]
                X_batch = X_tensor[batch_indices]
                
                # Generate two augmented views
                aug1, aug2 = self._augment(X_batch)
                
                # Online network forward pass (view 1)
                online_features1 = self.online_network(aug1)
                online_projection1 = self.online_projector(online_features1)
                online_prediction1 = self.online_predictor(online_projection1)
                
                # Online network forward pass (view 2)
                online_features2 = self.online_network(aug2)
                online_projection2 = self.online_projector(online_features2)
                online_prediction2 = self.online_predictor(online_projection2)
                
                # Target network forward pass (view 2)
                with torch.no_grad():
                    target_features2 = self.target_network(aug2)
                    target_projection2 = self.target_projector(target_features2)
                
                # Target network forward pass (view 1)
                with torch.no_grad():
                    target_features1 = self.target_network(aug1)
                    target_projection1 = self.target_projector(target_features1)
                
                # Compute loss (symmetric)
                loss1 = self._cosine_similarity_loss(online_prediction1, target_projection2)
                loss2 = self._cosine_similarity_loss(online_prediction2, target_projection1)
                loss = (loss1 + loss2) / 2
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update target network
                self._update_target_network(self.moving_average_decay)
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
        
        return self
    
    def get_representations(self, X):
        self.online_network.eval()
        self.online_projector.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            features = self.online_network(X_tensor)
            projections = self.online_projector(features)
            return projections.cpu().numpy()
    
    def get_params(self):
        return {
            'moving_average_decay': self.moving_average_decay,
            'projection_dim': self.projection_dim,
            'hidden_dim': self.hidden_dim,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate
        }