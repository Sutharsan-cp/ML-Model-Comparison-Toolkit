import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class VICReg:
    def __init__(self, encoder, projector_hidden_dim=2048, projector_output_dim=2048,
                 sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0, 
                 epochs=100, batch_size=256, learning_rate=0.001, device='cpu'):
        self.encoder = encoder
        self.projector_hidden_dim = projector_hidden_dim
        self.projector_output_dim = projector_output_dim
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        
        # Build projector
        self.projector = self._build_projector()
        
    def _build_projector(self):
        return nn.Sequential(
            nn.Linear(self.encoder.output_dim, self.projector_hidden_dim),
            nn.BatchNorm1d(self.projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.projector_hidden_dim, self.projector_hidden_dim),
            nn.BatchNorm1d(self.projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.projector_hidden_dim, self.projector_output_dim)
        )
    
    def _invariance_loss(self, z1, z2):
        """Invariance loss: MSE between representations"""
        return nn.MSELoss()(z1, z2)
    
    def _variance_loss(self, z1, z2):
        """Variance loss: encourage standard deviation to be above threshold"""
        std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
        std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
        
        # Hinge loss to encourage std above 1.0
        std_loss = torch.mean(torch.relu(1 - std_z1)) + torch.mean(torch.relu(1 - std_z2))
        return std_loss / 2
    
    def _covariance_loss(self, z1, z2):
        """Covariance loss: decorrelate features"""
        def _off_diagonal(x):
            n, m = x.shape
            assert n == m
            return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
        
        batch_size = z1.shape[0]
        
        # Center the representations
        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)
        
        # Compute covariance matrices
        cov_z1 = (z1.T @ z1) / (batch_size - 1)
        cov_z2 = (z2.T @ z2) / (batch_size - 1)
        
        # Off-diagonal elements should be close to 0
        cov_loss = _off_diagonal(cov_z1).pow_(2).sum() / self.projector_output_dim
        cov_loss += _off_diagonal(cov_z2).pow_(2).sum() / self.projector_output_dim
        
        return cov_loss / 2
    
    def _augment(self, x):
        """Create two augmented views"""
        aug1 = x + 0.1 * torch.randn_like(x)
        aug1 = aug1 * (0.8 + 0.4 * torch.rand(1).to(self.device))
        
        aug2 = x + 0.15 * torch.randn_like(x)
        aug2 = aug2 * (0.7 + 0.6 * torch.rand(1).to(self.device))
        
        return aug1, aug2
    
    def fit(self, X):
        X_tensor = torch.FloatTensor(X).to(self.device)
        n_samples = len(X_tensor)
        
        optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.projector.parameters()),
            lr=self.learning_rate, weight_decay=1e-6
        )
        
        for epoch in range(self.epochs):
            self.encoder.train()
            self.projector.train()
            
            total_loss = 0
            total_invariance_loss = 0
            total_variance_loss = 0
            total_covariance_loss = 0
            
            # Shuffle data
            indices = torch.randperm(n_samples)
            
            for i in range(0, n_samples, self.batch_size):
                batch_indices = indices[i:min(i+self.batch_size, n_samples)]
                X_batch = X_tensor[batch_indices]
                
                # Generate two augmented views
                aug1, aug2 = self._augment(X_batch)
                
                # Forward pass
                h1 = self.encoder(aug1)
                z1 = self.projector(h1)
                
                h2 = self.encoder(aug2)
                z2 = self.projector(h2)
                
                # Compute losses
                invariance_loss = self._invariance_loss(z1, z2)
                variance_loss = self._variance_loss(z1, z2)
                covariance_loss = self._covariance_loss(z1, z2)
                
                # Total loss
                loss = (self.sim_coeff * invariance_loss + 
                       self.std_coeff * variance_loss + 
                       self.cov_coeff * covariance_loss)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_invariance_loss += invariance_loss.item()
                total_variance_loss += variance_loss.item()
                total_covariance_loss += covariance_loss.item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}, "
                      f"Invariance: {total_invariance_loss:.4f}, "
                      f"Variance: {total_variance_loss:.4f}, "
                      f"Covariance: {total_covariance_loss:.4f}")
        
        return self
    
    def get_representations(self, X):
        self.encoder.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            representations = self.encoder(X_tensor)
            return representations.cpu().numpy()
    
    def get_projected_representations(self, X):
        self.encoder.eval()
        self.projector.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            h = self.encoder(X_tensor)
            z = self.projector(h)
            return z.cpu().numpy()
    
    def get_params(self):
        return {
            'projector_hidden_dim': self.projector_hidden_dim,
            'projector_output_dim': self.projector_output_dim,
            'sim_coeff': self.sim_coeff,
            'std_coeff': self.std_coeff,
            'cov_coeff': self.cov_coeff,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate
        }