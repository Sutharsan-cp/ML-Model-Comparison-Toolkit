import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class MeanTeacher:
    def __init__(self, student_model, teacher_model=None, alpha=0.99, 
                 consistency_weight=1.0, consistency_type='mse', 
                 epochs=100, batch_size=32, learning_rate=0.001, 
                 device='cpu'):
        self.student_model = student_model
        self.teacher_model = teacher_model if teacher_model is not None else self._copy_model(student_model)
        self.alpha = alpha
        self.consistency_weight = consistency_weight
        self.consistency_type = consistency_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        
        # Initialize teacher with student weights
        self._update_teacher(alpha=0.0)
        
    def _copy_model(self, model):
        """Create a copy of the model"""
        return type(model)(**model.get_config() if hasattr(model, 'get_config') else {})
    
    def _update_teacher(self, alpha=None):
        """Update teacher model with exponential moving average of student weights"""
        if alpha is None:
            alpha = self.alpha
        
        for teacher_param, student_param in zip(self.teacher_model.parameters(), 
                                              self.student_model.parameters()):
            teacher_param.data = alpha * teacher_param.data + (1 - alpha) * student_param.data
    
    def _consistency_loss(self, student_output, teacher_output):
        """Compute consistency loss between student and teacher predictions"""
        if self.consistency_type == 'mse':
            return nn.MSELoss()(student_output, teacher_output.detach())
        elif self.consistency_type == 'kl':
            return nn.KLDivLoss()(torch.log_softmax(student_output, dim=1), 
                                torch.softmax(teacher_output.detach(), dim=1))
        else:
            raise ValueError(f"Unsupported consistency type: {self.consistency_type}")
    
    def fit(self, X_labeled, y_labeled, X_unlabeled):
        # Convert to tensors
        X_labeled_tensor = torch.FloatTensor(X_labeled).to(self.device)
        y_labeled_tensor = torch.LongTensor(y_labeled).to(self.device)
        X_unlabeled_tensor = torch.FloatTensor(X_unlabeled).to(self.device)
        
        # Optimizer
        optimizer = optim.Adam(self.student_model.parameters(), lr=self.learning_rate)
        
        # Loss functions
        supervised_criterion = nn.CrossEntropyLoss()
        
        # Training loop
        n_labeled = len(X_labeled_tensor)
        n_unlabeled = len(X_unlabeled_tensor)
        
        for epoch in range(self.epochs):
            self.student_model.train()
            self.teacher_model.train()
            
            # Shuffle data
            labeled_indices = torch.randperm(n_labeled)
            unlabeled_indices = torch.randperm(n_unlabeled)
            
            total_loss = 0
            total_supervised_loss = 0
            total_consistency_loss = 0
            
            # Mini-batch training
            for i in range(0, max(n_labeled, n_unlabeled), self.batch_size):
                # Get labeled batch
                labeled_batch_idx = labeled_indices[i:min(i+self.batch_size, n_labeled)]
                X_labeled_batch = X_labeled_tensor[labeled_batch_idx]
                y_labeled_batch = y_labeled_tensor[labeled_batch_idx]
                
                # Get unlabeled batch
                unlabeled_batch_idx = unlabeled_indices[i:min(i+self.batch_size, n_unlabeled)]
                X_unlabeled_batch = X_unlabeled_tensor[unlabeled_batch_idx]
                
                # Forward pass for labeled data
                student_labeled_output = self.student_model(X_labeled_batch)
                supervised_loss = supervised_criterion(student_labeled_output, y_labeled_batch)
                
                # Forward pass for unlabeled data
                if len(X_unlabeled_batch) > 0:
                    # Apply augmentation (in practice, you'd use actual augmentations)
                    X_unlabeled_aug1 = X_unlabeled_batch + 0.1 * torch.randn_like(X_unlabeled_batch)
                    X_unlabeled_aug2 = X_unlabeled_batch + 0.1 * torch.randn_like(X_unlabeled_batch)
                    
                    # Student prediction on augmented data
                    student_unlabeled_output1 = self.student_model(X_unlabeled_aug1)
                    student_unlabeled_output2 = self.student_model(X_unlabeled_aug2)
                    
                    # Teacher prediction (no gradient)
                    with torch.no_grad():
                        teacher_unlabeled_output1 = self.teacher_model(X_unlabeled_aug1)
                        teacher_unlabeled_output2 = self.teacher_model(X_unlabeled_aug2)
                    
                    # Consistency loss
                    consistency_loss1 = self._consistency_loss(student_unlabeled_output1, teacher_unlabeled_output2)
                    consistency_loss2 = self._consistency_loss(student_unlabeled_output2, teacher_unlabeled_output1)
                    consistency_loss = (consistency_loss1 + consistency_loss2) / 2
                else:
                    consistency_loss = torch.tensor(0.0).to(self.device)
                
                # Total loss
                loss = supervised_loss + self.consistency_weight * consistency_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update teacher model
                self._update_teacher()
                
                total_loss += loss.item()
                total_supervised_loss += supervised_loss.item()
                total_consistency_loss += consistency_loss.item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}, "
                      f"Supervised: {total_supervised_loss:.4f}, "
                      f"Consistency: {total_consistency_loss:.4f}")
        
        return self
    
    def predict(self, X):
        self.teacher_model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.teacher_model(X_tensor)
            _, predictions = torch.max(outputs, 1)
            return predictions.cpu().numpy()
    
    def predict_proba(self, X):
        self.teacher_model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.teacher_model(X_tensor)
            return torch.softmax(outputs, dim=1).cpu().numpy()
    
    def get_params(self):
        return {
            'alpha': self.alpha,
            'consistency_weight': self.consistency_weight,
            'consistency_type': self.consistency_type,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate
        }