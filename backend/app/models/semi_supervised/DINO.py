import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class DINO:
    def __init__(self, student_network, teacher_network=None, 
                 output_dim=65536, warmup_teacher_temp=0.04, 
                 teacher_temp=0.07, warmup_teacher_temp_epochs=30,
                 center_momentum=0.9, epochs=100, batch_size=256,
                 learning_rate=0.001, device='cpu'):
        self.student_network = student_network
        self.teacher_network = teacher_network if teacher_network is not None else self._copy_network(student_network)
        self.output_dim = output_dim
        self.warmup_teacher_temp = warmup_teacher_temp
        self.teacher_temp = teacher_temp
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.center_momentum = center_momentum
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        
        # Add projection heads
        self.student_head = self._build_head(student_network.output_dim, output_dim)
        self.teacher_head = self._build_head(teacher_network.output_dim, output_dim)
        
        # Initialize teacher with student weights
        self._update_teacher(0.0)
        
        # Center for teacher output
        self.register_buffer('teacher_center', torch.zeros(1, output_dim))
        
    def _copy_network(self, network):
        """Create a copy of the network"""
        return type(network)(**network.get_config() if hasattr(network, 'get_config') else {})
    
    def _build_head(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, output_dim)
        )
    
    def _update_teacher(self, momentum):
        """Update teacher with exponential moving average of student"""
        for student_param, teacher_param in zip(self.student_network.parameters(), 
                                              self.teacher_network.parameters()):
            teacher_param.data = momentum * teacher_param.data + (1 - momentum) * student_param.data
        
        for student_param, teacher_param in zip(self.student_head.parameters(),
                                              self.teacher_head.parameters()):
            teacher_param.data = momentum * teacher_param.data + (1 - momentum) * student_param.data
    
    def _sinkhorn_knopp(self, logits, n_iters=3):
        """Sinkhorn-Knopp algorithm for optimal transport"""
        Q = torch.exp(logits / 0.05)  # Temperature for Sinkhorn
        Q /= torch.sum(Q)
        
        K, B = Q.shape
        
        for _ in range(n_iters):
            # Normalize rows
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K
            
            # Normalize columns
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B
        
        Q *= B
        return Q.T
    
    def _dino_loss(self, student_output, teacher_output, epoch):
        """Compute DINO loss"""
        # Adjust teacher temperature
        if epoch < self.warmup_teacher_temp_epochs:
            teacher_temp = self.warmup_teacher_temp + (self.teacher_temp - self.warmup_teacher_temp) * epoch / self.warmup_teacher_temp_epochs
        else:
            teacher_temp = self.teacher_temp
        
        # Softmax with temperature
        student_out = student_output / 0.1  # Student temperature fixed at 0.1
        teacher_out = (teacher_output - self.teacher_center) / teacher_temp
        
        student_probs = torch.softmax(student_out, dim=-1)
        teacher_probs = torch.softmax(teacher_out, dim=-1)
        
        # Cross-entropy loss
        loss = -torch.sum(teacher_probs * torch.log(student_probs + 1e-8), dim=-1).mean()
        return loss
    
    def _augment(self, x, global_crop=True):
        """Create augmented views"""
        if global_crop:
            # Global views: stronger augmentation
            aug = x * (0.4 + 0.6 * torch.rand(1).to(self.device))
            aug = aug + 0.2 * torch.randn_like(x)
        else:
            # Local views: weaker augmentation
            aug = x * (0.8 + 0.4 * torch.rand(1).to(self.device))
            aug = aug + 0.1 * torch.randn_like(x)
        
        return aug
    
    def fit(self, X):
        X_tensor = torch.FloatTensor(X).to(self.device)
        n_samples = len(X_tensor)
        
        optimizer = optim.Adam(
            list(self.student_network.parameters()) + list(self.student_head.parameters()),
            lr=self.learning_rate, weight_decay=1e-6
        )
        
        for epoch in range(self.epochs):
            self.student_network.train()
            self.teacher_network.train()
            self.student_head.train()
            self.teacher_head.train()
            
            total_loss = 0
            n_batches = 0
            
            # Shuffle data
            indices = torch.randperm(n_samples)
            
            for i in range(0, n_samples, self.batch_size):
                batch_indices = indices[i:min(i+self.batch_size, n_samples)]
                X_batch = X_tensor[batch_indices]
                
                # Generate multiple views (2 global, several local)
                global_view1 = self._augment(X_batch, global_crop=True)
                global_view2 = self._augment(X_batch, global_crop=True)
                local_view = self._augment(X_batch, global_crop=False)
                
                # Student forward pass (all views)
                student_global1 = self.student_head(self.student_network(global_view1))
                student_global2 = self.student_head(self.student_network(global_view2))
                student_local = self.student_head(self.student_network(local_view))
                
                # Teacher forward pass (only global views, no gradients)
                with torch.no_grad():
                    teacher_global1 = self.teacher_head(self.teacher_network(global_view1))
                    teacher_global2 = self.teacher_head(self.teacher_network(global_view2))
                
                # Compute losses between different views
                loss1 = self._dino_loss(student_global1, teacher_global2, epoch)
                loss2 = self._dino_loss(student_global2, teacher_global1, epoch)
                loss3 = self._dino_loss(student_local, teacher_global1, epoch)
                
                loss = (loss1 + loss2 + loss3) / 3
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update teacher with momentum
                self._update_teacher(0.996)  # Default DINO momentum
                
                # Update center
                with torch.no_grad():
                    teacher_output = torch.cat([teacher_global1, teacher_global2])
                    batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
                    self.teacher_center = self.center_momentum * self.teacher_center + (1 - self.center_momentum) * batch_center
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / n_batches if n_batches > 0 else 0
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        return self
    
    def get_representations(self, X):
        self.teacher_network.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            representations = self.teacher_network(X_tensor)
            return representations.cpu().numpy()
    
    def get_params(self):
        return {
            'output_dim': self.output_dim,
            'warmup_teacher_temp': self.warmup_teacher_temp,
            'teacher_temp': self.teacher_temp,
            'warmup_teacher_temp_epochs': self.warmup_teacher_temp_epochs,
            'center_momentum': self.center_momentum,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate
        }