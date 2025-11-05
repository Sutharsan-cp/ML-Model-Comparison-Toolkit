import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class VirtualAdversarialTraining:
    def __init__(self, model, xi=1e-6, eps=2.0, num_power_iterations=1,
                 alpha=1.0, epochs=100, batch_size=32, learning_rate=0.001, device='cpu'):
        self.model = model
        self.xi = xi
        self.eps = eps
        self.num_power_iterations = num_power_iterations
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        
    def _normalize(self, v):
        return v / (torch.norm(v, dim=1, keepdim=True) + 1e-8)
    
    def _generate_virtual_adversarial_perturbation(self, x, logits):
        """Generate virtual adversarial perturbation"""
        d = torch.randn_like(x, requires_grad=True)
        
        for _ in range(self.num_power_iterations):
            d = self.xi * self._normalize(d)
            d.requires_grad_()
            
            # Forward pass with perturbation
            logits_perturbed = self.model(x + d)
            
            # KL divergence between original and perturbed distributions
            p_original = torch.softmax(logits, dim=1)
            p_perturbed = torch.softmax(logits_perturbed, dim=1)
            
            kl_div = torch.sum(p_original * torch.log(p_original / (p_perturbed + 1e-8)), dim=1).mean()
            
            # Compute gradient wrt d
            kl_div.backward()
            
            d = d.grad.data
            self.model.zero_grad()
        
        return self.eps * self._normalize(d)
    
    def fit(self, X_labeled, y_labeled, X_unlabeled):
        X_labeled_tensor = torch.FloatTensor(X_labeled).to(self.device)
        y_labeled_tensor = torch.LongTensor(y_labeled).to(self.device)
        X_unlabeled_tensor = torch.FloatTensor(X_unlabeled).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        supervised_criterion = nn.CrossEntropyLoss()
        
        n_labeled = len(X_labeled_tensor)
        n_unlabeled = len(X_unlabeled_tensor)
        X_all = torch.cat([X_labeled_tensor, X_unlabeled_tensor], dim=0)
        
        for epoch in range(self.epochs):
            self.model.train()
            
            total_loss = 0
            total_supervised_loss = 0
            total_vat_loss = 0
            
            # Shuffle data
            indices = torch.randperm(len(X_all))
            labeled_indices = torch.randperm(n_labeled)
            
            for i in range(0, len(X_all), self.batch_size):
                batch_indices = indices[i:min(i+self.batch_size, len(X_all))]
                X_batch = X_all[batch_indices]
                
                # Supervised loss for labeled data
                supervised_loss = torch.tensor(0.0).to(self.device)
                if i < n_labeled:
                    labeled_batch_indices = labeled_indices[i:min(i+self.batch_size, n_labeled)]
                    X_labeled_batch = X_labeled_tensor[labeled_batch_indices]
                    y_labeled_batch = y_labeled_tensor[labeled_batch_indices]
                    
                    logits_labeled = self.model(X_labeled_batch)
                    supervised_loss = supervised_criterion(logits_labeled, y_labeled_batch)
                
                # Virtual Adversarial Loss for all data
                X_batch.requires_grad_()
                logits_original = self.model(X_batch)
                
                # Generate adversarial perturbation
                with torch.no_grad():
                    r_adv = self._generate_virtual_adversarial_perturbation(X_batch, logits_original)
                
                # Compute VAT loss
                logits_perturbed = self.model(X_batch + r_adv)
                
                p_original = torch.softmax(logits_original, dim=1).detach()
                p_perturbed = torch.softmax(logits_perturbed, dim=1)
                
                vat_loss = torch.sum(p_original * torch.log(p_original / (p_perturbed + 1e-8)), dim=1).mean()
                
                # Total loss
                loss = supervised_loss + self.alpha * vat_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_supervised_loss += supervised_loss.item()
                total_vat_loss += vat_loss.item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}, "
                      f"Supervised: {total_supervised_loss:.4f}, "
                      f"VAT: {total_vat_loss:.4f}")
        
        return self
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            logits = self.model(X_tensor)
            _, predictions = torch.max(logits, 1)
            return predictions.cpu().numpy()
    
    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            logits = self.model(X_tensor)
            return torch.softmax(logits, dim=1).cpu().numpy()
    
    def get_params(self):
        return {
            'xi': self.xi,
            'eps': self.eps,
            'num_power_iterations': self.num_power_iterations,
            'alpha': self.alpha,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate
        }