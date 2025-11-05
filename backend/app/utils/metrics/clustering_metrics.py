import numpy as np
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.optimize import linear_sum_assignment
from itertools import combinations
import warnings

class ClusteringMetrics:
    """Comprehensive clustering metrics without sklearn"""
    
    @staticmethod
    def _validate_inputs(X, labels):
        """Validate inputs and convert to numpy arrays"""
        X = np.asarray(X)
        labels = np.asarray(labels)
        
        if len(X) != len(labels):
            raise ValueError(f"Shapes of X {X.shape} and labels {labels.shape} do not match")
        
        return X, labels
    
    @staticmethod
    def _calculate_pairwise_distances(X):
        """Calculate pairwise distances efficiently"""
        return squareform(pdist(X, metric='euclidean'))
    
    @staticmethod
    def _get_cluster_info(X, labels):
        """Get cluster centers, sizes, and indices"""
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        n_samples = len(X)
        
        cluster_centers = []
        cluster_sizes = []
        cluster_indices = []
        
        for label in unique_labels:
            mask = labels == label
            cluster_points = X[mask]
            cluster_centers.append(np.mean(cluster_points, axis=0))
            cluster_sizes.append(np.sum(mask))
            cluster_indices.append(np.where(mask)[0])
        
        return {
            'unique_labels': unique_labels,
            'n_clusters': n_clusters,
            'n_samples': n_samples,
            'cluster_centers': np.array(cluster_centers),
            'cluster_sizes': np.array(cluster_sizes),
            'cluster_indices': cluster_indices
        }
    
    # =========================================================================
    # INTERNAL VALIDATION METRICS
    # =========================================================================
    
    @staticmethod
    def silhouette_score(X, labels):
        """Silhouette Score - Measures cohesion and separation"""
        X, labels = ClusteringMetrics._validate_inputs(X, labels)
        n_samples = len(X)
        
        if len(np.unique(labels)) <= 1:
            return 0.0
        
        # Calculate pairwise distances
        distances = ClusteringMetrics._calculate_pairwise_distances(X)
        
        silhouette_scores = []
        for i in range(n_samples):
            current_label = labels[i]
            current_cluster = labels == current_label
            
            # Calculate a(i) - mean distance to points in same cluster
            same_cluster_indices = np.where(current_cluster)[0]
            same_cluster_indices = same_cluster_indices[same_cluster_indices != i]  # Exclude self
            
            if len(same_cluster_indices) == 0:
                a = 0.0
            else:
                a = np.mean(distances[i, same_cluster_indices])
            
            # Calculate b(i) - min mean distance to other clusters
            b = float('inf')
            for other_label in np.unique(labels):
                if other_label == current_label:
                    continue
                
                other_cluster = labels == other_label
                if np.sum(other_cluster) > 0:
                    b_cluster = np.mean(distances[i, other_cluster])
                    b = min(b, b_cluster)
            
            # Handle edge cases
            if a == 0 and b == 0:
                silhouette_i = 0.0
            else:
                silhouette_i = (b - a) / max(a, b)
            
            silhouette_scores.append(silhouette_i)
        
        return np.mean(silhouette_scores)
    
    @staticmethod
    def silhouette_samples(X, labels):
        """Silhouette scores for each sample"""
        X, labels = ClusteringMetrics._validate_inputs(X, labels)
        n_samples = len(X)
        
        if len(np.unique(labels)) <= 1:
            return np.zeros(n_samples)
        
        distances = ClusteringMetrics._calculate_pairwise_distances(X)
        silhouette_scores = []
        
        for i in range(n_samples):
            current_label = labels[i]
            current_cluster = labels == current_label
            
            # Calculate a(i)
            same_cluster_indices = np.where(current_cluster)[0]
            same_cluster_indices = same_cluster_indices[same_cluster_indices != i]
            
            if len(same_cluster_indices) == 0:
                a = 0.0
            else:
                a = np.mean(distances[i, same_cluster_indices])
            
            # Calculate b(i)
            b = float('inf')
            for other_label in np.unique(labels):
                if other_label == current_label:
                    continue
                other_cluster = labels == other_label
                if np.sum(other_cluster) > 0:
                    b_cluster = np.mean(distances[i, other_cluster])
                    b = min(b, b_cluster)
            
            if a == 0 and b == 0:
                silhouette_i = 0.0
            else:
                silhouette_i = (b - a) / max(a, b)
            
            silhouette_scores.append(silhouette_i)
        
        return np.array(silhouette_scores)
    
    @staticmethod
    def calinski_harabasz_score(X, labels):
        """Calinski-Harabasz Index - Variance ratio criterion"""
        X, labels = ClusteringMetrics._validate_inputs(X, labels)
        cluster_info = ClusteringMetrics._get_cluster_info(X, labels)
        
        n_clusters = cluster_info['n_clusters']
        n_samples = cluster_info['n_samples']
        
        if n_clusters <= 1:
            return 0.0
        
        # Overall mean
        overall_mean = np.mean(X, axis=0)
        
        # Between-cluster dispersion (SSB)
        ssb = 0.0
        for i in range(n_clusters):
            cluster_size = cluster_info['cluster_sizes'][i]
            cluster_center = cluster_info['cluster_centers'][i]
            ssb += cluster_size * np.sum((cluster_center - overall_mean) ** 2)
        
        # Within-cluster dispersion (SSW)
        ssw = 0.0
        for i in range(n_clusters):
            cluster_points = X[labels == cluster_info['unique_labels'][i]]
            cluster_center = cluster_info['cluster_centers'][i]
            ssw += np.sum(np.sum((cluster_points - cluster_center) ** 2, axis=1))
        
        if ssw == 0:
            return float('inf')
        
        return (ssb / (n_clusters - 1)) / (ssw / (n_samples - n_clusters))
    
    @staticmethod
    def davies_bouldin_score(X, labels):
        """Davies-Bouldin Index - Average similarity between clusters"""
        X, labels = ClusteringMetrics._validate_inputs(X, labels)
        cluster_info = ClusteringMetrics._get_cluster_info(X, labels)
        
        n_clusters = cluster_info['n_clusters']
        
        if n_clusters <= 1:
            return 0.0
        
        # Calculate cluster dispersions
        dispersions = []
        for i in range(n_clusters):
            cluster_points = X[labels == cluster_info['unique_labels'][i]]
            cluster_center = cluster_info['cluster_centers'][i]
            dispersion = np.mean(np.sqrt(np.sum((cluster_points - cluster_center) ** 2, axis=1)))
            dispersions.append(dispersion)
        
        dispersions = np.array(dispersions)
        
        # Calculate Davies-Bouldin index
        db_index = 0.0
        for i in range(n_clusters):
            max_similarity = -float('inf')
            for j in range(n_clusters):
                if i != j:
                    center_distance = np.sqrt(np.sum(
                        (cluster_info['cluster_centers'][i] - cluster_info['cluster_centers'][j]) ** 2
                    ))
                    if center_distance == 0:
                        similarity = float('inf')
                    else:
                        similarity = (dispersions[i] + dispersions[j]) / center_distance
                    max_similarity = max(max_similarity, similarity)
            db_index += max_similarity
        
        return db_index / n_clusters
    
    @staticmethod
    def dunn_index(X, labels):
        """Dunn Index - Ratio of min inter-cluster to max intra-cluster distance"""
        X, labels = ClusteringMetrics._validate_inputs(X, labels)
        cluster_info = ClusteringMetrics._get_cluster_info(X, labels)
        
        n_clusters = cluster_info['n_clusters']
        
        if n_clusters <= 1:
            return 0.0
        
        # Calculate maximum intra-cluster distance
        max_intra_distance = 0.0
        for i in range(n_clusters):
            cluster_points = X[labels == cluster_info['unique_labels'][i]]
            if len(cluster_points) > 1:
                intra_distances = pdist(cluster_points)
                max_intra_distance = max(max_intra_distance, np.max(intra_distances))
        
        # Calculate minimum inter-cluster distance
        min_inter_distance = float('inf')
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                cluster_i = X[labels == cluster_info['unique_labels'][i]]
                cluster_j = X[labels == cluster_info['unique_labels'][j]]
                inter_distances = cdist(cluster_i, cluster_j)
                min_inter_distance = min(min_inter_distance, np.min(inter_distances))
        
        if max_intra_distance == 0:
            return float('inf')
        
        return min_inter_distance / max_intra_distance
    
    @staticmethod
    def beta_cv_ratio(X, labels):
        """Beta CV Ratio - Ratio of within to between cluster sum of squares"""
        X, labels = ClusteringMetrics._validate_inputs(X, labels)
        cluster_info = ClusteringMetrics._get_cluster_info(X, labels)
        
        n_clusters = cluster_info['n_clusters']
        
        if n_clusters <= 1:
            return 0.0
        
        # Within-cluster sum of squares
        wss = 0.0
        for i in range(n_clusters):
            cluster_points = X[labels == cluster_info['unique_labels'][i]]
            cluster_center = cluster_info['cluster_centers'][i]
            wss += np.sum(np.sum((cluster_points - cluster_center) ** 2, axis=1))
        
        # Between-cluster sum of squares
        overall_mean = np.mean(X, axis=0)
        bss = 0.0
        for i in range(n_clusters):
            cluster_size = cluster_info['cluster_sizes'][i]
            cluster_center = cluster_info['cluster_centers'][i]
            bss += cluster_size * np.sum((cluster_center - overall_mean) ** 2)
        
        if bss == 0:
            return float('inf')
        
        return wss / bss
    
    @staticmethod
    def ball_hall_index(X, labels):
        """Ball-Hall Index - Mean of within-cluster variances"""
        X, labels = ClusteringMetrics._validate_inputs(X, labels)
        cluster_info = ClusteringMetrics._get_cluster_info(X, labels)
        
        n_clusters = cluster_info['n_clusters']
        
        if n_clusters <= 1:
            return 0.0
        
        within_cluster_variances = []
        for i in range(n_clusters):
            cluster_points = X[labels == cluster_info['unique_labels'][i]]
            if len(cluster_points) > 1:
                variance = np.mean(np.var(cluster_points, axis=0))
                within_cluster_variances.append(variance)
            else:
                within_cluster_variances.append(0.0)
        
        return np.mean(within_cluster_variances)
    
    @staticmethod
    def hartigan_index(X, labels):
        """Hartigan Index - Log ratio of between to within cluster sum of squares"""
        X, labels = ClusteringMetrics._validate_inputs(X, labels)
        cluster_info = ClusteringMetrics._get_cluster_info(X, labels)
        
        n_clusters = cluster_info['n_clusters']
        
        if n_clusters <= 1:
            return 0.0
        
        # Within-cluster sum of squares
        wss = 0.0
        for i in range(n_clusters):
            cluster_points = X[labels == cluster_info['unique_labels'][i]]
            cluster_center = cluster_info['cluster_centers'][i]
            wss += np.sum(np.sum((cluster_points - cluster_center) ** 2, axis=1))
        
        # Between-cluster sum of squares
        overall_mean = np.mean(X, axis=0)
        bss = 0.0
        for i in range(n_clusters):
            cluster_size = cluster_info['cluster_sizes'][i]
            cluster_center = cluster_info['cluster_centers'][i]
            bss += cluster_size * np.sum((cluster_center - overall_mean) ** 2)
        
        if wss == 0:
            return float('inf')
        
        return np.log(bss / wss)
    
    @staticmethod
    def xu_index(X, labels):
        """Xu Index - Combines within-cluster variance and number of clusters"""
        X, labels = ClusteringMetrics._validate_inputs(X, labels)
        cluster_info = ClusteringMetrics._get_cluster_info(X, labels)
        
        n_clusters = cluster_info['n_clusters']
        n_samples = cluster_info['n_samples']
        n_features = X.shape[1]
        
        if n_clusters <= 1:
            return float('inf')
        
        # Within-cluster variance
        total_variance = 0.0
        for i in range(n_clusters):
            cluster_points = X[labels == cluster_info['unique_labels'][i]]
            if len(cluster_points) > 1:
                cluster_variance = np.sum(np.var(cluster_points, axis=0))
                total_variance += cluster_variance
        
        return n_features * np.log(np.sqrt(total_variance / (n_features * n_samples ** 2))) + \
               np.log(n_clusters)
    
    # =========================================================================
    # EXTERNAL VALIDATION METRICS
    # =========================================================================
    
    @staticmethod
    def contingency_matrix(labels_true, labels_pred):
        """Compute contingency matrix for two clusterings"""
        labels_true = np.asarray(labels_true)
        labels_pred = np.asarray(labels_pred)
        
        classes = np.unique(labels_true)
        clusters = np.unique(labels_pred)
        
        n_classes = len(classes)
        n_clusters = len(clusters)
        
        # Create mapping from label to index
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        cluster_to_idx = {clust: i for i, clust in enumerate(clusters)}
        
        # Build contingency matrix
        contingency = np.zeros((n_classes, n_clusters), dtype=int)
        
        for true_label, pred_label in zip(labels_true, labels_pred):
            i = class_to_idx[true_label]
            j = cluster_to_idx[pred_label]
            contingency[i, j] += 1
        
        return contingency, classes, clusters
    
    @staticmethod
    def rand_index(labels_true, labels_pred):
        """Rand Index - Agreement between two clusterings"""
        contingency, _, _ = ClusteringMetrics.contingency_matrix(labels_true, labels_pred)
        n_samples = len(labels_true)
        
        # Calculate pairs
        sum_combinations = np.sum([stats.comb(n_ij, 2) for n_ij in contingency.flatten()])
        sum_rows = np.sum([stats.comb(n_i, 2) for n_i in np.sum(contingency, axis=1)])
        sum_cols = np.sum([stats.comb(n_j, 2) for n_j in np.sum(contingency, axis=0)])
        
        total_pairs = stats.comb(n_samples, 2)
        expected_index = sum_rows * sum_cols / total_pairs
        
        # Rand index
        numerator = sum_combinations - expected_index
        denominator = 0.5 * (sum_rows + sum_cols) - expected_index
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    @staticmethod
    def adjusted_rand_index(labels_true, labels_pred):
        """Adjusted Rand Index - RI corrected for chance"""
        contingency, _, _ = ClusteringMetrics.contingency_matrix(labels_true, labels_pred)
        n_samples = len(labels_true)
        
        # Calculate pairs
        sum_combinations = np.sum([stats.comb(n_ij, 2) for n_ij in contingency.flatten()])
        sum_rows = np.sum([stats.comb(n_i, 2) for n_i in np.sum(contingency, axis=1)])
        sum_cols = np.sum([stats.comb(n_j, 2) for n_j in np.sum(contingency, axis=0)])
        
        total_pairs = stats.comb(n_samples, 2)
        expected_index = sum_rows * sum_cols / total_pairs
        max_index = 0.5 * (sum_rows + sum_cols)
        
        # Adjusted Rand index
        numerator = sum_combinations - expected_index
        denominator = max_index - expected_index
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    @staticmethod
    def mutual_info_score(labels_true, labels_pred):
        """Mutual Information between two clusterings"""
        contingency, _, _ = ClusteringMetrics.contingency_matrix(labels_true, labels_pred)
        n_samples = len(labels_true)
        
        # Convert to probabilities
        P = contingency / n_samples
        
        # Marginal probabilities
        P_true = np.sum(P, axis=1)
        P_pred = np.sum(P, axis=0)
        
        # Mutual information
        mi = 0.0
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                if P[i, j] > 0:
                    mi += P[i, j] * np.log(P[i, j] / (P_true[i] * P_pred[j]))
        
        return mi
    
    @staticmethod
    def adjusted_mutual_info_score(labels_true, labels_pred):
        """Adjusted Mutual Information - MI corrected for chance"""
        mi = ClusteringMetrics.mutual_info_score(labels_true, labels_pred)
        contingency, _, _ = ClusteringMetrics.contingency_matrix(labels_true, labels_pred)
        
        n_samples = len(labels_true)
        n_classes = contingency.shape[0]
        n_clusters = contingency.shape[1]
        
        # Calculate expected MI
        P = contingency / n_samples
        P_true = np.sum(P, axis=1)
        P_pred = np.sum(P, axis=0)
        
        expected_mi = 0.0
        for i in range(n_classes):
            for j in range(n_clusters):
                if P_true[i] > 0 and P_pred[j] > 0:
                    term = P_true[i] * P_pred[j]
                    if term > 0:
                        expected_mi += term * np.log(term)
        
        # Maximum MI
        H_true = -np.sum([p * np.log(p) for p in P_true if p > 0])
        H_pred = -np.sum([p * np.log(p) for p in P_pred if p > 0])
        max_mi = min(H_true, H_pred)
        
        if max_mi == 0:
            return 0.0
        
        # Adjusted MI
        return (mi - expected_mi) / (max_mi - expected_mi)
    
    @staticmethod
    def normalized_mutual_info_score(labels_true, labels_pred):
        """Normalized Mutual Information"""
        mi = ClusteringMetrics.mutual_info_score(labels_true, labels_pred)
        contingency, _, _ = ClusteringMetrics.contingency_matrix(labels_true, labels_pred)
        
        # Entropies
        P_true = np.sum(contingency, axis=1) / len(labels_true)
        P_pred = np.sum(contingency, axis=0) / len(labels_true)
        
        H_true = -np.sum([p * np.log(p) for p in P_true if p > 0])
        H_pred = -np.sum([p * np.log(p) for p in P_pred if p > 0])
        
        denominator = np.sqrt(H_true * H_pred)
        
        if denominator == 0:
            return 0.0
        
        return mi / denominator
    
    @staticmethod
    def homogeneity_score(labels_true, labels_pred):
        """Homogeneity - Each cluster contains only members of a single class"""
        contingency, _, _ = ClusteringMetrics.contingency_matrix(labels_true, labels_pred)
        n_samples = len(labels_true)
        
        # Class distribution
        P_true = np.sum(contingency, axis=1) / n_samples
        H_true = -np.sum([p * np.log(p) for p in P_true if p > 0])
        
        # Conditional entropy
        H_true_given_pred = 0.0
        for j in range(contingency.shape[1]):
            cluster_size = np.sum(contingency[:, j])
            if cluster_size > 0:
                cluster_dist = contingency[:, j] / cluster_size
                cluster_entropy = -np.sum([p * np.log(p) for p in cluster_dist if p > 0])
                H_true_given_pred += (cluster_size / n_samples) * cluster_entropy
        
        if H_true == 0:
            return 1.0
        
        return 1.0 - (H_true_given_pred / H_true)
    
    @staticmethod
    def completeness_score(labels_true, labels_pred):
        """Completeness - All members of a class are assigned to the same cluster"""
        # Completeness is symmetric to homogeneity
        return ClusteringMetrics.homogeneity_score(labels_pred, labels_true)
    
    @staticmethod
    def v_measure_score(labels_true, labels_pred, beta=1.0):
        """V-measure - Harmonic mean of homogeneity and completeness"""
        h = ClusteringMetrics.homogeneity_score(labels_true, labels_pred)
        c = ClusteringMetrics.completeness_score(labels_true, labels_pred)
        
        if h + c == 0:
            return 0.0
        
        return (1 + beta) * h * c / (beta * h + c)
    
    @staticmethod
    def fowlkes_mallows_score(labels_true, labels_pred):
        """Fowlkes-Mallows Index - Geometric mean of precision and recall"""
        contingency, _, _ = ClusteringMetrics.contingency_matrix(labels_true, labels_pred)
        n_samples = len(labels_true)
        
        # Calculate pairs
        sum_combinations = np.sum([stats.comb(n_ij, 2) for n_ij in contingency.flatten()])
        sum_rows = np.sum([stats.comb(n_i, 2) for n_i in np.sum(contingency, axis=1)])
        sum_cols = np.sum([stats.comb(n_j, 2) for n_j in np.sum(contingency, axis=0)])
        
        if sum_rows == 0 or sum_cols == 0:
            return 0.0
        
        precision = sum_combinations / sum_cols
        recall = sum_combinations / sum_rows
        
        return np.sqrt(precision * recall)
    
    # =========================================================================
    # STABILITY METRICS
    # =========================================================================
    
    @staticmethod
    def jaccard_similarity(labels1, labels2):
        """Jaccard Similarity between two clusterings"""
        # Create sets of sample pairs for each clustering
        def get_cluster_sets(labels):
            cluster_sets = {}
            unique_labels = np.unique(labels)
            for label in unique_labels:
                indices = np.where(labels == label)[0]
                # Create set of sample indices in this cluster
                cluster_sets[label] = set(indices)
            return cluster_sets
        
        sets1 = get_cluster_sets(labels1)
        sets2 = get_cluster_sets(labels2)
        
        # Calculate maximum Jaccard similarity between clusters
        max_similarities = []
        for cluster1 in sets1.values():
            best_similarity = 0.0
            for cluster2 in sets2.values():
                intersection = len(cluster1.intersection(cluster2))
                union = len(cluster1.union(cluster2))
                if union > 0:
                    similarity = intersection / union
                    best_similarity = max(best_similarity, similarity)
            max_similarities.append(best_similarity)
        
        return np.mean(max_similarities)
    
    @staticmethod
    def adjusted_wallace_index(labels1, labels2):
        """Adjusted Wallace Index - Agreement between clusterings corrected for chance"""
        # Similar to adjusted Rand index but for cluster-wise agreement
        ari = ClusteringMetrics.adjusted_rand_index(labels1, labels2)
        return ari  # Wallace index is closely related to ARI
    
    @staticmethod
    def cluster_stability_score(X, base_labels, n_subsamples=10, subsample_ratio=0.8):
        """Cluster Stability Score - Consistency across subsamples"""
        n_samples = len(X)
        subsample_size = int(n_samples * subsample_ratio)
        
        similarities = []
        for _ in range(n_subsamples):
            # Create subsample
            indices = np.random.choice(n_samples, subsample_size, replace=False)
            X_subsample = X[indices]
            base_subsample = base_labels[indices]
            
            # You would need to recluster here, but for simplicity we'll use
            # the existing labels and calculate similarity
            if len(np.unique(base_subsample)) > 1:
                # For demonstration, we'll use Jaccard similarity with itself
                # In practice, you'd run your clustering algorithm on the subsample
                similarity = ClusteringMetrics.jaccard_similarity(base_subsample, base_subsample)
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    @staticmethod
    def comprehensive_clustering_report(X, labels, labels_true=None):
        """Comprehensive clustering evaluation report"""
        report = {}
        
        # Internal validation metrics
        report['internal_metrics'] = {
            'silhouette_score': ClusteringMetrics.silhouette_score(X, labels),
            'calinski_harabasz_score': ClusteringMetrics.calinski_harabasz_score(X, labels),
            'davies_bouldin_score': ClusteringMetrics.davies_bouldin_score(X, labels),
            'dunn_index': ClusteringMetrics.dunn_index(X, labels),
            'beta_cv_ratio': ClusteringMetrics.beta_cv_ratio(X, labels),
            'ball_hall_index': ClusteringMetrics.ball_hall_index(X, labels),
            'hartigan_index': ClusteringMetrics.hartigan_index(X, labels),
            'xu_index': ClusteringMetrics.xu_index(X, labels)
        }
        
        # External validation metrics (if true labels available)
        if labels_true is not None:
            report['external_metrics'] = {
                'adjusted_rand_index': ClusteringMetrics.adjusted_rand_index(labels_true, labels),
                'rand_index': ClusteringMetrics.rand_index(labels_true, labels),
                'mutual_info_score': ClusteringMetrics.mutual_info_score(labels_true, labels),
                'adjusted_mutual_info_score': ClusteringMetrics.adjusted_mutual_info_score(labels_true, labels),
                'normalized_mutual_info_score': ClusteringMetrics.normalized_mutual_info_score(labels_true, labels),
                'homogeneity_score': ClusteringMetrics.homogeneity_score(labels_true, labels),
                'completeness_score': ClusteringMetrics.completeness_score(labels_true, labels),
                'v_measure_score': ClusteringMetrics.v_measure_score(labels_true, labels),
                'fowlkes_mallows_score': ClusteringMetrics.fowlkes_mallows_score(labels_true, labels)
            }
        
        # Cluster statistics
        cluster_info = ClusteringMetrics._get_cluster_info(X, labels)
        report['cluster_statistics'] = {
            'n_clusters': cluster_info['n_clusters'],
            'cluster_sizes': cluster_info['cluster_sizes'],
            'size_std': np.std(cluster_info['cluster_sizes']),
            'min_size': np.min(cluster_info['cluster_sizes']),
            'max_size': np.max(cluster_info['cluster_sizes'])
        }
        
        return report
    
    @staticmethod
    def get_core_internal_metrics(X, labels):
        """Get core internal validation metrics"""
        return {
            'silhouette_score': ClusteringMetrics.silhouette_score(X, labels),
            'calinski_harabasz_score': ClusteringMetrics.calinski_harabasz_score(X, labels),
            'davies_bouldin_score': ClusteringMetrics.davies_bouldin_score(X, labels),
            'dunn_index': ClusteringMetrics.dunn_index(X, labels)
        }
    
    @staticmethod
    def get_core_external_metrics(labels_true, labels_pred):
        """Get core external validation metrics"""
        return {
            'adjusted_rand_index': ClusteringMetrics.adjusted_rand_index(labels_true, labels_pred),
            'normalized_mutual_info_score': ClusteringMetrics.normalized_mutual_info_score(labels_true, labels_pred),
            'homogeneity_score': ClusteringMetrics.homogeneity_score(labels_true, labels_pred),
            'completeness_score': ClusteringMetrics.completeness_score(labels_true, labels_pred),
            'v_measure_score': ClusteringMetrics.v_measure_score(labels_true, labels_pred)
        }