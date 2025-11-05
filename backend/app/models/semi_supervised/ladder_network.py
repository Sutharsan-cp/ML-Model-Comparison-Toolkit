import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class LadderNetwork:
    def __init__(self, encoder_layers, decoder_layers, noise_std=0.3, 
                 supervised_weight=1.0, unsupervised_weight=1.0,
                 epochs=100, batch_size=32, learning_rate=0.001, device='cpu'):
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.noise_std = noise_std
        self.supervised_weight = supervised_weight
        self.unsupervised_weight = unsupervised_weight
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.classifier = nn.Linear(encoder_layers[-1], 2)  # Binary classification
        
    def _build_encoder(self):
        layers = []
        for i in range(len(self.encoder_layers) - 1):
            layers.append(nn.Linear(self.encoder_layers[i], self.encoder_layers[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(self.encoder_layers[i+1]))
        return nn.Sequential(*layers)
    
    def _build_decoder(self):
        layers = []
        for i in range(len(self.decoder_layers) - 1):
            layers.append(nn.Linear(self.decoder_layers[i], self.decoder_layers[i+1]))
            if i < len(self.decoder_layers) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm1d(self.decoder_layers[i+1]))
        return nn.Sequential(*layers)
    
    def _add_noise(self, x, noise_std):
        return x + torch.randn_like(x) * noise_std
    
    def _forward_pass(self, x_clean, x_noisy):
        # Encoder pass (clean and noisy paths)
        clean_activations = []
        h_clean = x_clean
        for layer in self.encoder:
            h_clean = layer(h_clean)
            if isinstance(layer, nn.ReLU):
                clean_activations.append(h_clean)
        
        noisy_activations = []
        h_noisy = x_noisy
        for layer in self.encoder:
            h_noisy = layer(h_noisy)
            if isinstance(layer, nn.ReLU):
                noisy_activations.append(h_noisy)
        
        # Classifier
        logits = self.classifier(h_clean)
        
        # Decoder pass (reconstruction)
        reconstructions = []
        h_dec = h_noisy
        for i, layer in enumerate(self.decoder):
            h_dec = layer(h_dec)
            if i % 3 == 0:  # After linear layer
                # Skip connection from encoder
                if len(noisy_activations) > 0:
                    enc_activation = noisy_activations.pop()
                    h_dec = h_dec + enc_activation
                reconstructions.append(h_dec)
        
        return logits, clean_activations, reconstructions
    
    def _compute_loss(self, logits, targets, clean_activations, reconstructions):
        # Supervised loss
        supervised_loss = nn.CrossEntropyLoss()(logits, targets)
        
        # Unsupervised loss (reconstruction)
        unsupervised_loss = 0
        for clean, recon in zip(clean_activations, reconstructions):
            # Ensure same shape
            min_dim = min(clean.shape[1], recon.shape[1])
            unsupervised_loss += nn.MSELoss()(clean[:, :min_dim], recon[:, :min_dim])
        
        total_loss = (self.supervised_weight * supervised_loss + 
                     self.unsupervised_weight * unsupervised_loss)
        
        return total_loss, supervised_loss, unsupervised_loss
    
    def fit(self, X_labeled, y_labeled, X_unlabeled):
        X_labeled_tensor = torch.FloatTensor(X_labeled).to(self.device)
        y_labeled_tensor = torch.LongTensor(y_labeled).to(self.device)
        X_unlabeled_tensor = torch.FloatTensor(X_unlabeled).to(self.device)
        
        # Combine all data for unsupervised learning
        X_all = torch.cat([X_labeled_tensor, X_unlabeled_tensor], dim=0)
        
        optimizer = optim.Adam(
            list(self.encoder.parameters()) + 
            list(self.decoder.parameters()) + 
            list(self.classifier.parameters()), 
            lr=self.learning_rate
        )
        
        n_labeled = len(X_labeled_tensor)
        n_all = len(X_all)
        
        for epoch in range(self.epochs):
            self.encoder.train()
            self.decoder.train()
            self.classifier.train()
            
            total_loss = 0
            total_supervised_loss = 0
            total_unsupervised_loss = 0
            
            # Shuffle data
            indices = torch.randperm(n_all)
            labeled_indices = torch.randperm(n_labeled)
            
            for i in range(0, n_all, self.batch_size):
                batch_indices = indices[i:min(i+self.batch_size, n_all)]
                X_batch = X_all[batch_indices]
                
                # Add noise for noisy path
                X_noisy = self._add_noise(X_batch, self.noise_std)
                
                # Forward pass
                logits, clean_activations, reconstructions = self._forward_pass(X_batch, X_noisy)
                
                # For labeled data, compute supervised loss
                if i < n_labeled:
                    labeled_batch_indices = labeled_indices[i:min(i+self.batch_size, n_labeled)]
                    targets = y_labeled_tensor[labeled_batch_indices]
                    
                    # Use only the first part of logits corresponding to labeled data
                    logits_labeled = logits[:len(targets)]
                    
                    loss, supervised_loss, unsupervised_loss = self._compute_loss(
                        logits_labeled, targets, clean_activations, reconstructions
                    )
                else:
                    # For unlabeled data, only unsupervised loss
                    supervised_loss = torch.tensor(0.0)
                    unsupervised_loss = self.unsupervised_weight * sum(
                        nn.MSELoss()(clean[:, :min(clean.shape[1], recon.shape[1])], 
                                    recon[:, :min(clean.shape[1], recon.shape[1])])
                        for clean, recon in zip(clean_activations, reconstructions)
                    )
                    loss = unsupervised_loss
                
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
        self.encoder.eval()
        self.classifier.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            clean_activations = []
            h = X_tensor
            for layer in self.encoder:
                h = layer(h)
            logits = self.classifier(h)
            _, predictions = torch.max(logits, 1)
            return predictions.cpu().numpy()
    
    def get_params(self):
        return {
            'encoder_layers': self.encoder_layers,
            'decoder_layers': self.decoder_layers,
            'noise_std': self.noise_std,
            'supervised_weight': self.supervised_weight,
            'unsupervised_weight': self.unsupervised_weight,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate
        }