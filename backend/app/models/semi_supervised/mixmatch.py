import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class MixMatch:
    def __init__(self, model, T=0.5, alpha=0.75, K=2, lambda_u=100,
                 epochs=100, batch_size=32, learning_rate=0.001, device='cpu'):
        self.model = model
        self.T = T  # Temperature for sharpening
        self.alpha = alpha  # Beta distribution parameter for mixup
        self.K = K  # Number of augmentations per unlabeled sample
        self.lambda_u = lambda_u  # Weight for unsupervised loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        
    def _sharpening(self, p, T):
        """Sharpen probability distribution"""
        p = p ** (1 / T)
        return p / p.sum(dim=1, keepdim=True)
    
    def _mixup(self, x1, x2, y1, y2, alpha=0.75):
        """Mixup data augmentation"""
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
        lam = max(lam, 1 - lam)  # Ensure lam >= 0.5
        
        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_y = lam * y1 + (1 - lam) * y2
        
        return mixed_x, mixed_y, lam
    
    def _guess_labels(self, X_unlabeled, model):
        """Guess labels for unlabeled data using model predictions"""
        with torch.no_grad():
            logits = model(X_unlabeled)
            p = torch.softmax(logits / self.T, dim=1)
            # Average over multiple augmentations in practice
            return p
    
    def fit(self, X_labeled, y_labeled, X_unlabeled):
        X_labeled_tensor = torch.FloatTensor(X_labeled).to(self.device)
        y_labeled_tensor = torch.LongTensor(y_labeled).to(self.device)
        X_unlabeled_tensor = torch.FloatTensor(X_unlabeled).to(self.device)
        
        # Convert labels to one-hot
        n_classes = len(torch.unique(y_labeled_tensor))
        y_labeled_onehot = torch.eye(n_classes).to(self.device)[y_labeled_tensor]
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
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
            
            for i in range(0, min(n_labeled, n_unlabeled), self.batch_size):
                # Get labeled batch
                labeled_batch_idx = labeled_indices[i:min(i+self.batch_size, n_labeled)]
                X_labeled_batch = X_labeled_tensor[labeled_batch_idx]
                y_labeled_batch = y_labeled_onehot[labeled_batch_idx]
                
                # Get unlabeled batch
                unlabeled_batch_idx = unlabeled_indices[i:min(i+self.batch_size, n_unlabeled)]
                X_unlabeled_batch = X_unlabeled_tensor[unlabeled_batch_idx]
                
                # Step 1: Guess labels for unlabeled data
                with torch.no_grad():
                    # In practice, you'd use K augmentations and average
                    guessed_probs = self._guess_labels(X_unlabeled_batch, self.model)
                    guessed_probs = self._sharpening(guessed_probs, self.T)
                
                # Step 2: MixUp
                # Combine labeled and unlabeled data
                X_combined = torch.cat([X_labeled_batch, X_unlabeled_batch], dim=0)
                y_combined = torch.cat([y_labeled_batch, guessed_probs], dim=0)
                
                # Shuffle for mixup
                indices = torch.randperm(len(X_combined))
                X_combined_shuffled = X_combined[indices]
                y_combined_shuffled = y_combined[indices]
                
                # Apply mixup to labeled data
                X_labeled_mixed, y_labeled_mixed, lam = self._mixup(
                    X_labeled_batch, X_combined_shuffled[:len(X_labeled_batch)],
                    y_labeled_batch, y_combined_shuffled[:len(X_labeled_batch)],
                    self.alpha
                )
                
                # Apply mixup to unlabeled data
                X_unlabeled_mixed, y_unlabeled_mixed, _ = self._mixup(
                    X_unlabeled_batch, X_combined_shuffled[len(X_labeled_batch):],
                    guessed_probs, y_combined_shuffled[len(X_labeled_batch):],
                    self.alpha
                )
                
                # Step 3: Compute losses
                # Supervised loss on mixed labeled data
                logits_labeled = self.model(X_labeled_mixed)
                supervised_loss = -torch.sum(y_labeled_mixed * torch.log_softmax(logits_labeled, dim=1)) / len(X_labeled_mixed)
                
                # Unsupervised loss on mixed unlabeled data
                logits_unlabeled = self.model(X_unlabeled_mixed)
                unsupervised_loss = torch.mean((torch.softmax(logits_unlabeled, dim=1) - y_unlabeled_mixed) ** 2)
                
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
            'T': self.T,
            'alpha': self.alpha,
            'K': self.K,
            'lambda_u': self.lambda_u,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate
        }