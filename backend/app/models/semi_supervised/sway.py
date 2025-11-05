import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class SWAY:
    def __init__(self, model, alpha=1.0, beta=1.0, temperature=0.1,
                 epochs=100, batch_size=32, learning_rate=0.001, device='cpu'):
        self.model = model
        self.alpha = alpha  # Weight for supervised loss
        self.beta = beta    # Weight for contrastive loss
        self.temperature = temperature
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        
    def _contrastive_loss(self, z1, z2, labels, temperature):
        """Compute supervised contrastive loss"""
        batch_size = z1.shape[0]
        
        # Normalize representations
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)
        
        # Combine representations
        z = torch.cat([z1, z2], dim=0)
        
        # Expand labels for both views
        labels = torch.cat([labels, labels], dim=0)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(z, z.T) / temperature
        
        # Create mask for positive pairs (same class)
        mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        
        # Remove diagonal
        mask = mask & ~torch.eye(2 * batch_size, dtype=torch.bool).to(self.device)
        
        # Compute positive and negative similarities
        positives = similarity_matrix[mask].view(2 * batch_size, -1)
        negatives = similarity_matrix[~mask].view(2 * batch_size, -1)
        
        # Compute logits
        logits = torch.cat([positives, negatives], dim=1)
        
        # Labels: first positives.size(1) columns are positive
        targets = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        
        loss = nn.CrossEntropyLoss()(logits, targets)
        return loss
    
    def _augment(self, x):
        """Create two augmented views"""
        aug1 = x + 0.1 * torch.randn_like(x)
        aug2 = x * (0.8 + 0.4 * torch.rand(1).to(self.device))
        return aug1, aug2
    
    def fit(self, X_labeled, y_labeled, X_unlabeled):
        X_labeled_tensor = torch.FloatTensor(X_labeled).to(self.device)
        y_labeled_tensor = torch.LongTensor(y_labeled).to(self.device)
        X_unlabeled_tensor = torch.FloatTensor(X_unlabeled).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        supervised_criterion = nn.CrossEntropyLoss()
        
        n_labeled = len(X_labeled_tensor)
        n_unlabeled = len(X_unlabeled_tensor)
        
        for epoch in range(self.epochs):
            self.model.train()
            
            total_loss = 0
            total_supervised_loss = 0
            total_contrastive_loss = 0
            
            # Process labeled and unlabeled data
            max_batches = max(n_labeled, n_unlabeled) // self.batch_size
            
            for batch_idx in range(max_batches):
                # Labeled data batch
                if batch_idx * self.batch_size < n_labeled:
                    start_idx = batch_idx * self.batch_size
                    end_idx = min((batch_idx + 1) * self.batch_size, n_labeled)
                    
                    X_labeled_batch = X_labeled_tensor[start_idx:end_idx]
                    y_labeled_batch = y_labeled_tensor[start_idx:end_idx]
                    
                    # Supervised loss
                    logits = self.model(X_labeled_batch)
                    supervised_loss = supervised_criterion(logits, y_labeled_batch)
                    
                    # Contrastive loss on labeled data
                    aug1, aug2 = self._augment(X_labeled_batch)
                    features1 = self.model.features(aug1) if hasattr(self.model, 'features') else self.model(aug1)
                    features2 = self.model.features(aug2) if hasattr(self.model, 'features') else self.model(aug2)
                    
                    contrastive_loss = self._contrastive_loss(
                        features1, features2, y_labeled_batch, self.temperature
                    )
                else:
                    supervised_loss = torch.tensor(0.0).to(self.device)
                    contrastive_loss = torch.tensor(0.0).to(self.device)
                
                # Unlabeled data for contrastive learning
                if batch_idx * self.batch_size < n_unlabeled:
                    start_idx = batch_idx * self.batch_size
                    end_idx = min((batch_idx + 1) * self.batch_size, n_unlabeled)
                    
                    X_unlabeled_batch = X_unlabeled_tensor[start_idx:end_idx]
                    
                    # Generate pseudo-labels for unlabeled data
                    with torch.no_grad():
                        logits_unlabeled = self.model(X_unlabeled_batch)
                        pseudo_labels = torch.argmax(logits_unlabeled, dim=1)
                    
                    # Contrastive loss on unlabeled data with pseudo-labels
                    aug1, aug2 = self._augment(X_unlabeled_batch)
                    features1 = self.model.features(aug1) if hasattr(self.model, 'features') else self.model(aug1)
                    features2 = self.model.features(aug2) if hasattr(self.model, 'features') else self.model(aug2)
                    
                    unlabeled_contrastive_loss = self._contrastive_loss(
                        features1, features2, pseudo_labels, self.temperature
                    )
                    
                    contrastive_loss = (contrastive_loss + unlabeled_contrastive_loss) / 2
                
                # Total loss
                loss = self.alpha * supervised_loss + self.beta * contrastive_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_supervised_loss += supervised_loss.item()
                total_contrastive_loss += contrastive_loss.item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}, "
                      f"Supervised: {total_supervised_loss:.4f}, "
                      f"Contrastive: {total_contrastive_loss:.4f}")
        
        return self
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            logits = self.model(X_tensor)
            _, predictions = torch.max(logits, 1)
            return predictions.cpu().numpy()
    
    def get_representations(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            if hasattr(self.model, 'features'):
                representations = self.model.features(X_tensor)
            else:
                representations = self.model(X_tensor)
            return representations.cpu().numpy()
    
    def get_params(self):
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'temperature': self.temperature,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate
        }