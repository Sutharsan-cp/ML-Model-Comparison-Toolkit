import numpy as np
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.linalg import eigh, svd, qr
from scipy.optimize import linear_sum_assignment
import warnings

class DimensionalityReductionMetrics:
    """Comprehensive dimensionality reduction evaluation metrics"""
    
    @staticmethod
    def _validate_inputs(X, X_reduced):
        """Validate inputs and convert to numpy arrays"""
        X = np.asarray(X)
        X_reduced = np.asarray(X_reduced)
        
        if len(X) != len(X_reduced):
            raise ValueError(f"Shapes of X {X.shape} and X_reduced {X_reduced.shape} do not match")
        
        return X, X_reduced
    
    @staticmethod
    def _calculate_pairwise_distances(X, metric='euclidean'):
        """Calculate pairwise distances efficiently"""
        return squareform(pdist(X, metric=metric))
    
    @staticmethod
    def _find_neighbors(distances, n_neighbors):
        """Find nearest neighbors for each point"""
        n_samples = len(distances)
        neighbors = np.zeros((n_samples, n_neighbors), dtype=int)
        
        for i in range(n_samples):
            # Get indices of nearest neighbors (excluding self)
            indices = np.argsort(distances[i])[1:n_neighbors+1]
            neighbors[i] = indices
        
        return neighbors
    
    # =========================================================================
    # PCA/ICA SPECIFIC METRICS
    # =========================================================================
    
    @staticmethod
    def explained_variance_ratio(X, X_reduced, method='pca'):
        """
        Explained variance ratio for PCA
        For ICA, measures how much variance is captured
        """
        X, X_reduced = DimensionalityReductionMetrics._validate_inputs(X, X_reduced)
        
        total_variance = np.var(X, axis=0).sum()
        
        if method.lower() == 'pca':
            # For PCA, calculate variance in reduced space
            reduced_variance = np.var(X_reduced, axis=0).sum()
        elif method.lower() == 'ica':
            # For ICA, reconstruction quality
            # Simple approach: use reconstruction error
            reconstructed = DimensionalityReductionMetrics._reconstruct_ica(X, X_reduced)
            reconstruction_variance = np.var(reconstructed, axis=0).sum()
            reduced_variance = reconstruction_variance
        else:
            raise ValueError("Method must be 'pca' or 'ica'")
        
        return reduced_variance / total_variance if total_variance > 0 else 0.0
    
    @staticmethod
    def _reconstruct_ica(X, components):
        """Simple ICA reconstruction (pseudo-inverse)"""
        # This is simplified - actual ICA reconstruction depends on the algorithm
        pinv_components = np.linalg.pinv(components.T)
        reconstructed = X @ pinv_components.T
        return reconstructed
    
    @staticmethod
    def cumulative_explained_variance(X, n_components=None):
        """
        Cumulative explained variance for PCA
        Useful for elbow method
        """
        X = np.asarray(X)
        
        if n_components is None:
            n_components = min(X.shape)
        
        # Center the data
        X_centered = X - np.mean(X, axis=0)
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)
        
        # Eigen decomposition
        eigenvalues, eigenvectors = eigh(cov_matrix)
        
        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        
        # Cumulative explained variance
        total_variance = np.sum(eigenvalues)
        explained_variance_ratio = eigenvalues / total_variance
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        return {
            'explained_variance_ratio': explained_variance_ratio[:n_components],
            'cumulative_variance': cumulative_variance[:n_components],
            'eigenvalues': eigenvalues[:n_components]
        }
    
    @staticmethod
    def kurtosis_ica(X_reduced):
        """
        Kurtosis for ICA evaluation
        ICA aims to maximize non-Gaussianity (high kurtosis)
        """
        X_reduced = np.asarray(X_reduced)
        
        kurtosis_values = []
        for i in range(X_reduced.shape[1]):
            component = X_reduced[:, i]
            # Standardize the component
            component_std = (component - np.mean(component)) / np.std(component)
            kurt = stats.kurtosis(component_std)
            kurtosis_values.append(kurt)
        
        return {
            'component_kurtosis': kurtosis_values,
            'mean_kurtosis': np.mean(kurtosis_values),
            'max_kurtosis': np.max(kurtosis_values),
            'min_kurtosis': np.min(kurtosis_values)
        }
    
    @staticmethod
    def mutual_information_ica(X_reduced):
        """
        Mutual information between components for ICA
        Good ICA should have minimal mutual information
        """
        X_reduced = np.asarray(X_reduced)
        n_components = X_reduced.shape[1]
        
        mi_matrix = np.zeros((n_components, n_components))
        
        for i in range(n_components):
            for j in range(i + 1, n_components):
                mi = DimensionalityReductionMetrics._calculate_mutual_info(
                    X_reduced[:, i], X_reduced[:, j]
                )
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi
        
        return {
            'mutual_information_matrix': mi_matrix,
            'mean_mutual_information': np.mean(mi_matrix[np.triu_indices_from(mi_matrix, k=1)]),
            'max_mutual_information': np.max(mi_matrix[np.triu_indices_from(mi_matrix, k=1)])
        }
    
    @staticmethod
    def _calculate_mutual_info(x, y, bins=20):
        """Calculate mutual information between two variables"""
        hist_xy, _, _ = np.histogram2d(x, y, bins=bins)
        hist_x, _ = np.histogram(x, bins=bins)
        hist_y, _ = np.histogram(y, bins=bins)
        
        # Convert to probabilities
        p_xy = hist_xy / np.sum(hist_xy)
        p_x = hist_x / np.sum(hist_x)
        p_y = hist_y / np.sum(hist_y)
        
        mi = 0.0
        for i in range(len(p_x)):
            for j in range(len(p_y)):
                if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
        
        return mi
    
    @staticmethod
    def pca_reconstruction_error(X, n_components):
        """
        PCA reconstruction error
        Measures how well data can be reconstructed from reduced representation
        """
        X = np.asarray(X)
        
        # Center the data
        X_centered = X - np.mean(X, axis=0)
        
        # Perform SVD
        U, s, Vt = svd(X_centered, full_matrices=False)
        
        # Reconstruct with n_components
        X_reconstructed = U[:, :n_components] @ np.diag(s[:n_components]) @ Vt[:n_components, :]
        
        # Calculate reconstruction error
        reconstruction_error = np.mean((X_centered - X_reconstructed) ** 2)
        
        return {
            'reconstruction_error': reconstruction_error,
            'reconstruction_rmse': np.sqrt(reconstruction_error),
            'explained_variance_ratio': np.sum(s[:n_components] ** 2) / np.sum(s ** 2)
        }
    
    # =========================================================================
    # t-SNE/UMAP SPECIFIC METRICS
    # =========================================================================
    
    @staticmethod
    def trustworthiness(X, X_reduced, n_neighbors=5, metric='euclidean'):
        """
        Trustworthiness measure
        Measures preservation of local structure
        """
        X, X_reduced = DimensionalityReductionMetrics._validate_inputs(X, X_reduced)
        n_samples = len(X)
        
        if n_neighbors >= n_samples:
            n_neighbors = n_samples - 1
        
        # Calculate distances in original and reduced space
        dist_original = DimensionalityReductionMetrics._calculate_pairwise_distances(X, metric)
        dist_reduced = DimensionalityReductionMetrics._calculate_pairwise_distances(X_reduced, metric)
        
        # Find neighbors in both spaces
        neighbors_original = DimensionalityReductionMetrics._find_neighbors(dist_original, n_neighbors)
        neighbors_reduced = DimensionalityReductionMetrics._find_neighbors(dist_reduced, n_neighbors)
        
        trust = 0.0
        for i in range(n_samples):
            # Neighbors in reduced space that are not neighbors in original space
            intruders = set(neighbors_reduced[i]) - set(neighbors_original[i])
            
            for j in intruders:
                # Find rank of j in original space
                original_ranks = np.argsort(dist_original[i])
                rank_j_original = np.where(original_ranks == j)[0][0]
                
                trust += max(0, rank_j_original - n_neighbors)
        
        trust = 1 - (2 * trust) / (n_samples * n_neighbors * (2 * n_samples - 3 * n_neighbors - 1))
        
        return max(0.0, min(1.0, trust))
    
    @staticmethod
    def continuity(X, X_reduced, n_neighbors=5, metric='euclidean'):
        """
        Continuity measure
        Complementary to trustworthiness - measures missing neighbors
        """
        X, X_reduced = DimensionalityReductionMetrics._validate_inputs(X, X_reduced)
        n_samples = len(X)
        
        if n_neighbors >= n_samples:
            n_neighbors = n_samples - 1
        
        # Calculate distances in original and reduced space
        dist_original = DimensionalityReductionMetrics._calculate_pairwise_distances(X, metric)
        dist_reduced = DimensionalityReductionMetrics._calculate_pairwise_distances(X_reduced, metric)
        
        # Find neighbors in both spaces
        neighbors_original = DimensionalityReductionMetrics._find_neighbors(dist_original, n_neighbors)
        neighbors_reduced = DimensionalityReductionMetrics._find_neighbors(dist_reduced, n_neighbors)
        
        continuity_val = 0.0
        for i in range(n_samples):
            # Neighbors in original space that are not neighbors in reduced space
            missing = set(neighbors_original[i]) - set(neighbors_reduced[i])
            
            for j in missing:
                # Find rank of j in reduced space
                reduced_ranks = np.argsort(dist_reduced[i])
                rank_j_reduced = np.where(reduced_ranks == j)[0][0]
                
                continuity_val += max(0, rank_j_reduced - n_neighbors)
        
        continuity_val = 1 - (2 * continuity_val) / (n_samples * n_neighbors * (2 * n_samples - 3 * n_neighbors - 1))
        
        return max(0.0, min(1.0, continuity_val))
    
    @staticmethod
    def neighborhood_preservation(X, X_reduced, n_neighbors=5, metric='euclidean'):
        """
        Neighborhood preservation score
        Measures overlap between neighborhoods in original and reduced space
        """
        X, X_reduced = DimensionalityReductionMetrics._validate_inputs(X, X_reduced)
        n_samples = len(X)
        
        if n_neighbors >= n_samples:
            n_neighbors = n_samples - 1
        
        dist_original = DimensionalityReductionMetrics._calculate_pairwise_distances(X, metric)
        dist_reduced = DimensionalityReductionMetrics._calculate_pairwise_distances(X_reduced, metric)
        
        neighbors_original = DimensionalityReductionMetrics._find_neighbors(dist_original, n_neighbors)
        neighbors_reduced = DimensionalityReductionMetrics._find_neighbors(dist_reduced, n_neighbors)
        
        preservation_scores = []
        for i in range(n_samples):
            overlap = len(set(neighbors_original[i]) & set(neighbors_reduced[i]))
            preservation_scores.append(overlap / n_neighbors)
        
        return {
            'mean_preservation': np.mean(preservation_scores),
            'std_preservation': np.std(preservation_scores),
            'min_preservation': np.min(preservation_scores),
            'max_preservation': np.max(preservation_scores)
        }
    
    @staticmethod
    def stress(X, X_reduced, metric='euclidean'):
        """
        Stress measure (Sammon's mapping)
        Measures preservation of pairwise distances
        """
        X, X_reduced = DimensionalityReductionMetrics._validate_inputs(X, X_reduced)
        
        dist_original = pdist(X, metric=metric)
        dist_reduced = pdist(X_reduced, metric=metric)
        
        # Avoid division by zero
        dist_original_safe = np.where(dist_original == 0, 1e-10, dist_original)
        
        stress_val = np.sum((dist_original - dist_reduced) ** 2 / dist_original_safe)
        stress_val /= np.sum(dist_original)
        
        return stress_val
    
    # =========================================================================
    # GENERAL DIMENSIONALITY REDUCTION METRICS
    # =========================================================================
    
    @staticmethod
    def reconstruction_error(X, X_reduced, method='auto'):
        """
        General reconstruction error
        Measures how well original data can be reconstructed
        """
        X, X_reduced = DimensionalityReductionMetrics._validate_inputs(X, X_reduced)
        
        if method == 'auto':
            # Simple linear reconstruction (pseudo-inverse)
            reconstruction_matrix = np.linalg.lstsq(X_reduced, X, rcond=None)[0]
            X_reconstructed = X_reduced @ reconstruction_matrix
        elif method == 'pca':
            # PCA-specific reconstruction
            X_centered = X - np.mean(X, axis=0)
            X_reduced_centered = X_reduced - np.mean(X_reduced, axis=0)
            reconstruction_matrix = np.linalg.lstsq(X_reduced_centered, X_centered, rcond=None)[0]
            X_reconstructed = X_reduced_centered @ reconstruction_matrix + np.mean(X, axis=0)
        else:
            raise ValueError("Method must be 'auto' or 'pca'")
        
        reconstruction_error = np.mean((X - X_reconstructed) ** 2)
        
        return {
            'reconstruction_error': reconstruction_error,
            'reconstruction_rmse': np.sqrt(reconstruction_error),
            'reconstruction_mae': np.mean(np.abs(X - X_reconstructed))
        }
    
    @staticmethod
    def local_structure_preservation(X, X_reduced, n_neighbors=10, metric='euclidean'):
        """
        Local structure preservation
        Measures how well local neighborhoods are preserved
        """
        trust = DimensionalityReductionMetrics.trustworthiness(X, X_reduced, n_neighbors, metric)
        cont = DimensionalityReductionMetrics.continuity(X, X_reduced, n_neighbors, metric)
        
        return {
            'trustworthiness': trust,
            'continuity': cont,
            'local_preservation_score': (trust + cont) / 2
        }
    
    @staticmethod
    def global_structure_preservation(X, X_reduced, metric='euclidean'):
        """
        Global structure preservation
        Measures how well global distances are preserved
        """
        X, X_reduced = DimensionalityReductionMetrics._validate_inputs(X, X_reduced)
        
        # Calculate correlation between distance matrices
        dist_original = pdist(X, metric=metric)
        dist_reduced = pdist(X_reduced, metric=metric)
        
        # Spearman correlation (rank-based, more robust)
        correlation = stats.spearmanr(dist_original, dist_reduced)[0]
        
        # Distance preservation ratio
        distance_ratio = dist_reduced / (dist_original + 1e-10)
        distance_preservation = 1 - np.std(distance_ratio)
        
        return {
            'distance_correlation': correlation,
            'distance_preservation': distance_preservation,
            'stress': DimensionalityReductionMetrics.stress(X, X_reduced, metric)
        }
    
    @staticmethod
    def kl_divergence_embedding(X, X_reduced, perplexity=30.0):
        """
        KL divergence for t-SNE like evaluation
        Measures how well probability distributions are preserved
        """
        X, X_reduced = DimensionalityReductionMetrics._validate_inputs(X, X_reduced)
        
        # Calculate pairwise distances
        dist_original = DimensionalityReductionMetrics._calculate_pairwise_distances(X)
        dist_reduced = DimensionalityReductionMetrics._calculate_pairwise_distances(X_reduced)
        
        # Convert to probabilities (similar to t-SNE)
        P = DimensionalityReductionMetrics._distance_to_probability(dist_original, perplexity)
        Q = DimensionalityReductionMetrics._distance_to_probability(dist_reduced, perplexity=1.0)  # Different perplexity
        
        # Calculate KL divergence
        kl_divergence = np.sum(P * np.log(P / Q))
        
        return kl_divergence
    
    @staticmethod
    def _distance_to_probability(distances, perplexity):
        """Convert distances to probability distributions (t-SNE style)"""
        n_samples = len(distances)
        
        # Calculate conditional probabilities
        P = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            # Find sigma that gives desired perplexity
            sigma = DimensionalityReductionMetrics._find_sigma(distances[i], perplexity)
            
            # Calculate probabilities
            P_i = np.exp(-distances[i] ** 2 / (2 * sigma ** 2))
            P_i[i] = 0  # Set self-probability to 0
            P_i = P_i / np.sum(P_i)
            
            P[i] = P_i
        
        # Symmetrize probabilities
        P = (P + P.T) / (2 * n_samples)
        
        return P
    
    @staticmethod
    def _find_sigma(distances_i, perplexity, tol=1e-5, max_iter=50):
        """Binary search for sigma that gives desired perplexity"""
        def perplexity_func(sigma):
            P = np.exp(-distances_i ** 2 / (2 * sigma ** 2))
            P = P / np.sum(P)
            entropy = -np.sum(P * np.log2(P + 1e-10))
            return 2 ** entropy
        
        sigma_min, sigma_max = 1e-10, 1000.0
        
        for _ in range(max_iter):
            sigma = (sigma_min + sigma_max) / 2
            current_perplexity = perplexity_func(sigma)
            
            if abs(current_perplexity - perplexity) < tol:
                break
            elif current_perplexity < perplexity:
                sigma_min = sigma
            else:
                sigma_max = sigma
        
        return sigma
    
    @staticmethod
    def cluster_preservation(X, X_reduced, labels, metric='euclidean'):
        """
        Measure how well cluster structure is preserved
        """
        X, X_reduced = DimensionalityReductionMetrics._validate_inputs(X, X_reduced)
        labels = np.asarray(labels)
        
        # Calculate silhouette scores in both spaces
        from clustering_metrics import ClusteringMetrics  # Assuming you have this class
        
        sil_original = ClusteringMetrics.silhouette_score(X, labels)
        sil_reduced = ClusteringMetrics.silhouette_score(X_reduced, labels)
        
        # Calculate within-cluster distances
        wcss_original = DimensionalityReductionMetrics._within_cluster_sum_of_squares(X, labels)
        wcss_reduced = DimensionalityReductionMetrics._within_cluster_sum_of_squares(X_reduced, labels)
        
        # Calculate between-cluster distances
        bcss_original = DimensionalityReductionMetrics._between_cluster_sum_of_squares(X, labels)
        bcss_reduced = DimensionalityReductionMetrics._between_cluster_sum_of_squares(X_reduced, labels)
        
        return {
            'silhouette_original': sil_original,
            'silhouette_reduced': sil_reduced,
            'silhouette_preservation': sil_reduced / sil_original if sil_original > 0 else 0,
            'wcss_ratio': wcss_reduced / wcss_original if wcss_original > 0 else float('inf'),
            'bcss_ratio': bcss_reduced / bcss_original if bcss_original > 0 else float('inf'),
            'separation_ratio': (bcss_reduced / wcss_reduced) / (bcss_original / wcss_original) if (wcss_original > 0 and wcss_reduced > 0) else 0
        }
    
    @staticmethod
    def _within_cluster_sum_of_squares(X, labels):
        """Calculate within-cluster sum of squares"""
        unique_labels = np.unique(labels)
        wcss = 0.0
        
        for label in unique_labels:
            cluster_points = X[labels == label]
            cluster_center = np.mean(cluster_points, axis=0)
            wcss += np.sum(np.sum((cluster_points - cluster_center) ** 2, axis=1))
        
        return wcss
    
    @staticmethod
    def _between_cluster_sum_of_squares(X, labels):
        """Calculate between-cluster sum of squares"""
        unique_labels = np.unique(labels)
        overall_center = np.mean(X, axis=0)
        bcss = 0.0
        
        for label in unique_labels:
            cluster_points = X[labels == label]
            cluster_center = np.mean(cluster_points, axis=0)
            bcss += len(cluster_points) * np.sum((cluster_center - overall_center) ** 2)
        
        return bcss
    
    # =========================================================================
    # COMPREHENSIVE EVALUATION
    # =========================================================================
    
    @staticmethod
    def comprehensive_dr_report(X, X_reduced, method='general', labels=None, n_neighbors=10):
        """
        Comprehensive dimensionality reduction evaluation report
        """
        report = {}
        
        # General metrics
        report['general_metrics'] = {
            'reconstruction_error': DimensionalityReductionMetrics.reconstruction_error(X, X_reduced),
            'local_structure': DimensionalityReductionMetrics.local_structure_preservation(X, X_reduced, n_neighbors),
            'global_structure': DimensionalityReductionMetrics.global_structure_preservation(X, X_reduced)
        }
        
        # Method-specific metrics
        if method.lower() == 'pca':
            report['pca_metrics'] = {
                'explained_variance_ratio': DimensionalityReductionMetrics.explained_variance_ratio(X, X_reduced, 'pca'),
                'cumulative_variance': DimensionalityReductionMetrics.cumulative_explained_variance(X),
                'reconstruction_error_pca': DimensionalityReductionMetrics.pca_reconstruction_error(X, X_reduced.shape[1])
            }
        
        elif method.lower() == 'ica':
            report['ica_metrics'] = {
                'explained_variance_ratio': DimensionalityReductionMetrics.explained_variance_ratio(X, X_reduced, 'ica'),
                'kurtosis': DimensionalityReductionMetrics.kurtosis_ica(X_reduced),
                'mutual_information': DimensionalityReductionMetrics.mutual_information_ica(X_reduced)
            }
        
        elif method.lower() in ['tsne', 'umap']:
            report['manifold_metrics'] = {
                'trustworthiness': DimensionalityReductionMetrics.trustworthiness(X, X_reduced, n_neighbors),
                'continuity': DimensionalityReductionMetrics.continuity(X, X_reduced, n_neighbors),
                'neighborhood_preservation': DimensionalityReductionMetrics.neighborhood_preservation(X, X_reduced, n_neighbors),
                'stress': DimensionalityReductionMetrics.stress(X, X_reduced)
            }
        
        # Cluster preservation (if labels available)
        if labels is not None:
            report['cluster_preservation'] = DimensionalityReductionMetrics.cluster_preservation(X, X_reduced, labels)
        
        # Dimensionality statistics
        report['dimensionality_stats'] = {
            'original_dimensions': X.shape[1],
            'reduced_dimensions': X_reduced.shape[1],
            'compression_ratio': X.shape[1] / X_reduced.shape[1],
            'samples_count': len(X)
        }
        
        return report
    
    @staticmethod
    def get_core_dr_metrics(X, X_reduced, n_neighbors=10):
        """Get core dimensionality reduction metrics for quick evaluation"""
        local_structure = DimensionalityReductionMetrics.local_structure_preservation(X, X_reduced, n_neighbors)
        global_structure = DimensionalityReductionMetrics.global_structure_preservation(X, X_reduced)
        reconstruction = DimensionalityReductionMetrics.reconstruction_error(X, X_reduced)
        
        return {
            'trustworthiness': local_structure['trustworthiness'],
            'continuity': local_structure['continuity'],
            'distance_correlation': global_structure['distance_correlation'],
            'reconstruction_rmse': reconstruction['reconstruction_rmse'],
            'stress': global_structure['stress']
        }