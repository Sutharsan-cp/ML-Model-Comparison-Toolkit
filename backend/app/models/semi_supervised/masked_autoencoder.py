import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class MaskedAutoencoder:
    def __init__(self, encoder, decoder, mask_ratio=0.75, 
                 epochs=100, batch_size=256, learning_rate=0.001, device='cpu'):
        self.encoder = encoder
        self.decoder = decoder
        self.mask_ratio = mask_ratio
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        
    def _random_masking(self, x, mask_ratio):
        """Randomly mask patches of the input"""
        batch_size, seq_len, dim = x.shape
        
        # Calculate number of tokens to mask
        num_masked = int(seq_len * mask_ratio)
        
        # Create random noise and get indices to mask
        noise = torch.rand(batch_size, seq_len, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Get masked indices
        ids_keep = ids_shuffle[:, :seq_len - num_masked]
        ids_mask = ids_shuffle[:, seq_len - num_masked:]
        
        # Keep unmasked tokens
        x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))
        
        # Generate binary mask: 0 is keep, 1 is remove
        mask = torch.ones(batch_size, seq_len, device=x.device)
        mask[:, :seq_len - num_masked] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_keep, ids_restore, mask, ids_keep, ids_mask
    
    def _patchify(self, x, patch_size):
        """Convert input to patches"""
        batch_size, channels, height, width = x.shape
        
        # Calculate number of patches
        num_patches_h = height // patch_size
        num_patches_w = width // patch_size
        num_patches = num_patches_h * num_patches_w
        patch_dim = channels * patch_size * patch_size
        
        # Create patches
        patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(batch_size, channels, num_patches_h, num_patches_w, patch_size, patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(batch_size, num_patches, patch_dim)
        
        return patches
    
    def _unpatchify(self, patches, patch_size, img_size):
        """Convert patches back to image"""
        batch_size, num_patches, patch_dim = patches.shape
        channels = patch_dim // (patch_size * patch_size)
        height = width = int(num_patches ** 0.5) * patch_size
        
        # Reshape to image
        patches = patches.view(batch_size, int(num_patches ** 0.5), int(num_patches ** 0.5), 
                             channels, patch_size, patch_size)
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = patches.view(batch_size, channels, height, width)
        
        return x
    
    def fit(self, X):
        # Assume X is already in image format (batch, channels, height, width)
        X_tensor = torch.FloatTensor(X).to(self.device)
        n_samples = len(X_tensor)
        
        optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.learning_rate, weight_decay=1e-6
        )
        
        reconstruction_criterion = nn.MSELoss()
        
        # Assuming image data with patches
        patch_size = 16  # Typical patch size for MAE
        img_size = int(X.shape[-1])  # Assuming square images
        
        for epoch in range(self.epochs):
            self.encoder.train()
            self.decoder.train()
            
            total_loss = 0
            total_reconstruction_loss = 0
            
            # Shuffle data
            indices = torch.randperm(n_samples)
            
            for i in range(0, n_samples, self.batch_size):
                batch_indices = indices[i:min(i+self.batch_size, n_samples)]
                X_batch = X_tensor[batch_indices]
                
                # Convert to patches
                patches = self._patchify(X_batch, patch_size)
                batch_size, num_patches, patch_dim = patches.shape
                
                # Normalize patches
                patches = patches - patches.mean(dim=-1, keepdim=True)
                patches = patches / (patches.std(dim=-1, keepdim=True) + 1e-6)
                
                # Apply random masking
                patches_keep, ids_restore, mask, ids_keep, ids_mask = self._random_masking(
                    patches, self.mask_ratio
                )
                
                # Encoder forward pass (only visible patches)
                latent = self.encoder(patches_keep)
                
                # Decoder forward pass (all patches)
                # Add mask tokens for masked patches
                mask_tokens = torch.zeros(batch_size, len(ids_mask[0]), latent.shape[-1]).to(self.device)
                full_latent = torch.cat([latent, mask_tokens], dim=1)
                
                # Restore original order
                full_latent = torch.gather(full_latent, dim=1, 
                                         index=ids_restore.unsqueeze(-1).repeat(1, 1, latent.shape[-1]))
                
                # Decode
                reconstruction = self.decoder(full_latent)
                
                # Compute loss only on masked patches
                target = patches
                reconstruction_loss = reconstruction_criterion(
                    reconstruction[torch.where(mask == 1)], 
                    target[torch.where(mask == 1)]
                )
                
                # Backward pass
                optimizer.zero_grad()
                reconstruction_loss.backward()
                optimizer.step()
                
                total_loss += reconstruction_loss.item()
                total_reconstruction_loss += reconstruction_loss.item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}, "
                      f"Reconstruction: {total_reconstruction_loss:.4f}")
        
        return self
    
    def get_representations(self, X):
        self.encoder.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            # Convert to patches
            patch_size = 16
            patches = self._patchify(X_tensor, patch_size)
            
            # Normalize
            patches = patches - patches.mean(dim=-1, keepdim=True)
            patches = patches / (patches.std(dim=-1, keepdim=True) + 1e-6)
            
            # Use all patches for representation
            representations = self.encoder(patches)
            return representations.cpu().numpy()
    
    def reconstruct(self, X):
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            patch_size = 16
            img_size = int(X.shape[-1])
            patches = self._patchify(X_tensor, patch_size)
            
            # Normalize
            patches = patches - patches.mean(dim=-1, keepdim=True)
            patches = patches / (patches.std(dim=-1, keepdim=True) + 1e-6)
            
            # Apply masking
            patches_keep, ids_restore, mask, _, _ = self._random_masking(patches, self.mask_ratio)
            
            # Encode and decode
            latent = self.encoder(patches_keep)
            
            # Add mask tokens
            batch_size, num_patches, patch_dim = patches.shape
            mask_tokens = torch.zeros(batch_size, int(num_patches * self.mask_ratio), 
                                    latent.shape[-1]).to(self.device)
            full_latent = torch.cat([latent, mask_tokens], dim=1)
            full_latent = torch.gather(full_latent, dim=1, 
                                     index=ids_restore.unsqueeze(-1).repeat(1, 1, latent.shape[-1]))
            
            reconstruction = self.decoder(full_latent)
            
            # Convert back to image
            reconstructed_img = self._unpatchify(reconstruction, patch_size, img_size)
            return reconstructed_img.cpu().numpy()
    
    def get_params(self):
        return {
            'mask_ratio': self.mask_ratio,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate
        }