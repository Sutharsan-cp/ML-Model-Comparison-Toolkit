import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class SimCLR:
    def __init__(self, encoder, projection_dim=128, temperature=0.5,
                 epochs=100, batch_size=256, learning_rate=0.001, device='cpu'):
        self.encoder = encoder
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        
        # Add projection head
        self.projector = nn.Sequential(
            nn.Linear(encoder.output_dim, encoder.output_dim),
            nn.ReLU(),
            nn.Linear(encoder.output_dim, projection_dim)
        )
        
    def _nt_xent_loss(self, z_i, z_j, temperature):
        """Compute NT-Xent loss (Normalized Temperature-scaled Cross Entropy)"""
        batch_size = z_i.shape[0]
        
        # Concatenate representations
        z = torch.cat([z_i, z_j], dim=0)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(z, z.T) / temperature
        
        # Create labels for positive pairs
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)
        
        # Remove diagonal (self-similarity)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        
        # Select positive and negative samples
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        
        # Compute logits
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        
        # Cross entropy loss
        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss
    
    def _augment(self, x):
        """Apply augmentations to create two views"""
        # View 1: Add noise and scale
        aug1 = x + 0.1 * torch.randn_like(x)
        aug1 = aug1 * (0.8 + 0.4 * torch.rand(1).to(self.device))
        
        # View 2: Different noise and scale
        aug2 = x + 0.15 * torch.randn_like(x)
        aug2 = aug2 * (0.7 + 0.6 * torch.rand(1).to(self.device))
        
        return aug1, aug2
    
    def fit(self, X):
        X_tensor = torch.FloatTensor(X).to(self.device)
        n_samples = len(X_tensor)
        
        optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.projector.parameters()),
            lr=self.learning_rate
        )
        
        for epoch in range(self.epochs):
            self.encoder.train()
            self.projector.train()
            
            total_loss = 0
            n_batches = 0
            
            # Shuffle data
            indices = torch.randperm(n_samples)
            
            for i in range(0, n_samples, self.batch_size):
                batch_indices = indices[i:min(i+self.batch_size, n_samples)]
                X_batch = X_tensor[batch_indices]
                
                # Generate two augmented views
                aug1, aug2 = self._augment(X_batch)
                
                # Forward pass for first view
                h1 = self.encoder(aug1)
                z1 = self.projector(h1)
                z1 = nn.functional.normalize(z1, dim=1)
                
                # Forward pass for second view
                h2 = self.encoder(aug2)
                z2 = self.projector(h2)
                z2 = nn.functional.normalize(z2, dim=1)
                
                # Compute contrastive loss
                loss = self._nt_xent_loss(z1, z2, self.temperature)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / n_batches if n_batches > 0 else 0
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
        
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
            z = nn.functional.normalize(z, dim=1)
            return z.cpu().numpy()
    
    def get_params(self):
        return {
            'projection_dim': self.projection_dim,
            'temperature': self.temperature,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate
        }