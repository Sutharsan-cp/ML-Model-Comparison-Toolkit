import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class FixMatch:
    def __init__(self, model, threshold=0.95, lambda_u=1.0, 
                 weak_augmentation=None, strong_augmentation=None,
                 epochs=100, batch_size=32, learning_rate=0.001, device='cpu'):
        self.model = model
        self.threshold = threshold
        self.lambda_u = lambda_u
        self.weak_augmentation = weak_augmentation
        self.strong_augmentation = strong_augmentation
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        
    def _apply_weak_augmentation(self, x):
        """Apply weak augmentation (e.g., random flip)"""
        if self.weak_augmentation is not None:
            return self.weak_augmentation(x)
        # Default: add small noise
        return x + 0.1 * torch.randn_like(x)
    
    def _apply_strong_augmentation(self, x):
        """Apply strong augmentation"""
        if self.strong_augmentation is not None:
            return self.strong_augmentation(x)
        # Default: add larger noise and scaling
        x_aug = x + 0.3 * torch.randn_like(x)
        scale = 0.8 + 0.4 * torch.rand(1).to(self.device)
        return x_aug * scale
    
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
            total_unsupervised_loss = 0
            
            # Shuffle data
            labeled_indices = torch.randperm(n_labeled)
            unlabeled_indices = torch.randperm(n_unlabeled)
            
            # Process in batches
            for i in range(0, min(n_labeled, n_unlabeled), self.batch_size):
                # Labeled batch
                labeled_batch_idx = labeled_indices[i:min(i+self.batch_size, n_labeled)]
                X_labeled_batch = X_labeled_tensor[labeled_batch_idx]
                y_labeled_batch = y_labeled_tensor[labeled_batch_idx]
                
                # Unlabeled batch
                unlabeled_batch_idx = unlabeled_indices[i:min(i+self.batch_size, n_unlabeled)]
                X_unlabeled_batch = X_unlabeled_tensor[unlabeled_batch_idx]
                
                # Supervised loss on labeled data
                logits_labeled = self.model(X_labeled_batch)
                supervised_loss = supervised_criterion(logits_labeled, y_labeled_batch)
                
                # Unsupervised loss on unlabeled data
                if len(X_unlabeled_batch) > 0:
                    # Weak augmentation for pseudo-labeling
                    X_unlabeled_weak = self._apply_weak_augmentation(X_unlabeled_batch)
                    
                    with torch.no_grad():
                        logits_weak = self.model(X_unlabeled_weak)
                        probs_weak = torch.softmax(logits_weak, dim=1)
                        max_probs, pseudo_labels = torch.max(probs_weak, dim=1)
                    
                    # Mask for high-confidence predictions
                    mask = max_probs >= self.threshold
                    
                    if mask.sum() > 0:
                        # Strong augmentation for consistency
                        X_unlabeled_strong = self._apply_strong_augmentation(X_unlabeled_batch[mask])
                        logits_strong = self.model(X_unlabeled_strong)
                        
                        # Cross-entropy with pseudo-labels
                        unsupervised_loss = nn.CrossEntropyLoss()(logits_strong, pseudo_labels[mask])
                    else:
                        unsupervised_loss = torch.tensor(0.0).to(self.device)
                else:
                    unsupervised_loss = torch.tensor(0.0).to(self.device)
                
                # Total loss
                loss = supervised_loss + self.lambda_u * unsupervised_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_supervised_loss += supervised_loss.item()
                total_unsupervised_loss += unsupervised_loss.item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}, "
                      f"Supervised: {total_supervised_loss:.4f}, "
                      f"Unsupervised: {total_unsupervised_loss:.4f}")
        
        return self
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            logits = self.model(X_tensor)
            _, predictions = torch.max(logits, 1)
            return predictions.cpu().numpy()
    
    def get_params(self):
        return {
            'threshold': self.threshold,
            'lambda_u': self.lambda_u,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate
        }