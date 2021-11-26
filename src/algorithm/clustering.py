import torch
import torch.nn as nn

class KMeansBase(nn.Module):
    def __init__(self, K=2, init_centroids='kmeans++'):
        super().__init__()

        self.K = K
        self.init_centroids = init_centroids

        self.cluster_ids, self.centroids = None, None
    
    def forward(self, data):
        """
        Args:
            data <torch.Tensor>: (batch_size, num_samples, num_features) or (num_samples, num_features)
        Returns:
            cluster_ids <torch.Tensor>: (batch_size, num_samples) or (num_samples,)
        """
        n_dims = data.dim()

        if n_dims == 2:
            data = data.unsqueeze(dim=0) # (batch_size, num_samples, num_features), where batch_size = 1.
        
        if self.training:
            if self.cluster_ids is None or self.centroids is None:
                cluster_ids, centroids = self._init_kmeans(data)
                self.cluster_ids, self.centroids = cluster_ids, centroids
        else:
            raise NotImplementedError
        
        if n_dims == 2:
            cluster_ids = cluster_ids.squeeze(dim=0) # (num_samples,)
            centroids = centroids.squeeze(dim=0) # (K, num_features)

            self.cluster_ids, self.centroids = cluster_ids, centroids

        return cluster_ids

    def _init_kmeans(self, data):
        K = self.K
        batch_size, num_samples, num_features = data.size()

        if self.init_centroids == 'kmeans++':
            centroid_ids = init_kmeans_pp(data, K=K) # (batch_size, K)
        else:
            centroid_ids = init_centroids_random(data, K=K) # (batch_size, K)
        
        centroid_ids  = centroid_ids.view(batch_size * K) # (batch_size * K)
        flatten_data = data.view(batch_size * num_samples, num_features) # (batch_size * num_samples, num_features)
        flatten_centroids = flatten_data[centroid_ids]
        centroids = flatten_centroids.view(batch_size, K, num_features) # (batch_size, K, num_features)
        
        distance = self.compute_distance(data.unsqueeze(dim=2), centroids.unsqueeze(dim=1), dim=3) # (batch_size, num_samples, K)
        cluster_ids = torch.argmin(distance, dim=2) # (batch_size, num_samples)

        return cluster_ids, centroids
    
    def compute_distance(self, x, y, dim=-1, keepdim=False):
        distance = torch.norm(x - y, dim=dim, keepdim=keepdim)

        return distance

class KMeans(KMeansBase):
    def __init__(self, K=2, init_centroids='kmeans++'):
        """
        Args:
            K <int>: number of clusters
        """
        super().__init__(K=K, init_centroids=init_centroids)
    
    def forward(self, data, iteration=None):
        """
        Args:
            data <torch.Tensor>: (batch_size, num_samples, num_features) or (num_samples, num_features)
        Returns:
            cluster_ids <torch.Tensor>: (batch_size, num_samples) or (num_samples,)
        """
        n_dims = data.dim()

        if n_dims == 2:
            data = data.unsqueeze(dim=0) # (batch_size, num_samples, num_features), where batch_size = 1.
        
        if self.training:
            if self.cluster_ids is None or self.centroids is None:
                self.cluster_ids, self.centroids = self._init_kmeans(data)
            
            if iteration is not None:
                for idx in range(iteration):
                    cluster_ids, centroids = self.update_once(data, cluster_ids=self.cluster_ids, centroids=self.centroids)

                    self.cluster_ids, self.centroids = cluster_ids, centroids
            else:
                while True:
                    cluster_ids, centroids = self.update_once(data, cluster_ids=self.cluster_ids, centroids=self.centroids)
                    distance = self.compute_distance(self.centroids, centroids, dim=-1)
                    distance = distance.mean().item()

                    self.cluster_ids, self.centroids = cluster_ids, centroids

                    if distance == 0:
                        break
        else:
            cluster_ids = self.infer(data)
            centroids = self.centroids
        
        if n_dims == 2:
            cluster_ids = cluster_ids.squeeze(dim=0) # (num_samples,)
            centroids = centroids.squeeze(dim=0) # (K, num_features)

            self.cluster_ids, self.centroids = cluster_ids, centroids

        return self.cluster_ids
        
    def update_once(self, data, cluster_ids=None, centroids=None):
        """
        Args:
            data: (batch_size, num_samples, num_features)
            cluster_ids: (batch_size, num_samples)
            centroids: (batch_size, K, num_features)
        Returns:
            cluster_ids: (batch_size, num_samples)
            centroids: (batch_size, K, num_features)
        """
        K = self.K
        mask = torch.eye(K)[cluster_ids] # (batch_size, num_samples, K)
        mask = mask.to(data.device)

        """
        1. Calculate centroids
        """
        masked_data = mask.unsqueeze(dim=3) * data.unsqueeze(dim=2) # (batch_size, num_samples, K, num_features)
        pseudo_centroids = masked_data.sum(dim=1) # (batch_size, K, num_features)
        denominator = mask.sum(dim=1).unsqueeze(dim=2) # (batch_size, K, 1)
        centroids = pseudo_centroids / denominator # (batch_size, K, num_features)
        
        """
        2. Put labels based on distance
        """
        distance = self.compute_distance(data.unsqueeze(dim=2), centroids.unsqueeze(dim=1), dim=3) # (batch_size, num_samples, K)
        cluster_ids = torch.argmin(distance, dim=2) # (batch_size, num_samples)
    
        return cluster_ids, centroids

    def infer(self, data):
        """
        Args:
            data <torch.Tensor>: (batch_size, num_samples, num_features)
        Returns:
            cluster_ids <torch.Tensor>: (batch_size, num_samples)
        """
        distance = self.compute_distance(data.unsqueeze(dim=2), self.centroids.unsqueeze(dim=1), dim=3) # (batch_size, num_samples, K)
        cluster_ids = torch.argmin(distance, dim=2) # (batch_size, num_samples)

        return cluster_ids

class GMMCluteringBase(nn.Module):
    def __init__(self, K=2, init_centroids='kmeans++', init_kmeans=True, diag_cov=False):
        super().__init__()

        self.K = K

        self.init_centroids, self.init_kmeans = init_centroids, init_kmeans
        self.diag_cov = diag_cov

        self.centroids, self.cov_matrix, self.mix_coeff = None, None, None
        self.cluster_probs = None

        if self.init_kmeans:
            self.kmeans = KMeans(K=K, init_centroids=init_centroids)

    def forward(self, data, centroids=None):
        """
        Args:
            data <torch.Tensor>: (batch_size, num_samples, num_features) or (num_samples, num_features)
            centroids <torch.Tensor>: (batch_size, K, num_features) or (K, num_features)
        Returns:
            cluster_ids <torch.Tensor>: (batch_size, num_samples) or (num_samples)
            centroids <torch.Tensor>: (batch_size, K, num_features) or (K, num_features)
        """
        n_dims = data.dim()

        if n_dims == 2:
            data = data.unsqueeze(dim=0) # (batch_size, num_samples, num_features), where batch_size = 1.
        
        if self.training:
            if self.cluster_probs is None or self.centroids is None or self.mix_coeff is None:
                centroids, cov_matrix, mix_coeff = self._init_GMM(data)
                self.centroids, self.cov_matrix, self.mix_coeff = centroids, cov_matrix, mix_coeff
        else:
            raise NotImplementedError
        
        cluster_probs = None
        
        if n_dims == 2:
            cluster_probs = cluster_probs.squeeze(dim=0) # (num_samples, K)
            centroids = centroids.squeeze(dim=0) # (K, num_features)

        self.cluster_probs, self.centroids = cluster_probs, centroids
        cluster_ids = torch.argmax(cluster_probs, dim=1) # (num_samples,)

        return cluster_ids

    def _init_GMM(self, data):
        K = self.K
        batch_size, num_samples, num_features = data.size()

        if self.init_kmeans:
            assert self.init_centroids == self.kmeans.init_centroids, "Invalid init_centroids is specified."
            _, centroids = self.kmeans(data)
        else:
            if self.init_centroids == 'kmeans++':
                centroid_ids = init_kmeans_pp(data, K=K) # (batch_size, K)
            else:
                centroid_ids = init_centroids_random(data, K=K) # (batch_size, K)

            centroid_ids  = centroid_ids.view(batch_size * K) # (batch_size * K)
            flatten_data = data.view(batch_size * num_samples, num_features) # (batch_size * num_samples, num_features)
            flatten_centroids = flatten_data[centroid_ids]
            centroids = flatten_centroids.view(batch_size, K, num_features) # (batch_size, K, num_features)
        
        cov_matrix, mix_coeff = torch.eye(K), torch.ones(K) / K # (K,), (K, K)
        cov_matrix, mix_coeff = torch.tile(cov_matrix, (batch_size, 1)), torch.tile(mix_coeff, (batch_size, 1, 1)) # (batch_size, K), (batch_size, K, K)
        cov_matrix, mix_coeff = cov_matrix.to(data.device), mix_coeff.to(data.device)
        
        return centroids, cov_matrix, mix_coeff

class GMMClustering(GMMCluteringBase):
    """
        Clustering based on Guassian Mixture Model.
    """
    def __init__(self, K=2, init_centroids='kmeans++', init_kmeans=True):
        super().__init__(K=K, init_centroids=init_centroids, init_kmeans=init_kmeans)

    def forward(self, data, iteration=None):
        """
        Args:
            data <torch.Tensor>: (batch_size, num_samples, num_features) or (num_samples, num_features)
        Returns:
            cluster_ids <torch.Tensor>: (batch_size, num_samples) or (num_samples,)
        """
        n_dims = data.dim()

        if n_dims == 2:
            data = data.unsqueeze(dim=0) # (batch_size, num_samples, num_features), where batch_size = 1.
        
        if self.training:
            if self.cluster_probs is None or self.centroids is None:
                self.centroids, self.cov_matrix, self.mix_coeff = self._init_GMM(data)

            if iteration is not None:
                for idx in range(iteration):
                    cluster_probs, centroids, cov_matrix, mix_coeff = self.update_once(data, cluster_probs=self.cluster_probs, centroids=self.centroids)

                    self.cluster_probs = cluster_probs
                    self.centroids, self.cov_matrix, self.mix_coeff = centroids, cov_matrix, mix_coeff
            else:
                while True:
                    cluster_probs, centroids, cov_matrix, mix_coeff = self.update_once(data, cluster_probs=self.cluster_probs, centroids=self.centroids)
                    distance = self.compute_distance(self.centroids, centroids, dim=-1)
                    distance = distance.mean().item()

                    self.cluster_probs = cluster_probs
                    self.centroids, self.cov_matrix, self.mix_coeff = centroids, cov_matrix, mix_coeff

                    if distance == 0:
                        break
        else:
            cluster_ids = self.infer(data)
            centroids = self.centroids
        
        if n_dims == 2:
            cluster_probs = cluster_probs.squeeze(dim=0) # (num_samples,)
            centroids, cov_matrix, mix_coeff = centroids.squeeze(dim=0).squeeze(dim=0), mix_coeff.squeeze(dim=0) # (K, num_features), (K, num_features, num_features), (K,)
            self.cluster_probs = cluster_probs
            self.centroids, self.cov_matrix, self.mix_coeff = centroids, cov_matrix, mix_coeff

            cluster_ids = torch.argmax(self.cluster_probs, dim=1) # (num_samples,)
        else:
            cluster_ids = torch.argmax(self.cluster_probs, dim=2) # (batch_size, num_samples)

        return cluster_ids
        
    def update_once(self, data, cluster_probs=None, centroids=None, cov_matrix=None, mix_coeff=None):
        """
        Args:
            data: (batch_size, num_samples, num_features)
            cluster_probs: (batch_size, num_samples, K)
            centroids: (batch_size, K, num_features)
            cov_matrix: (batch_size, K, num_features, num_features)
            mix_coeff: (batch_size, K)
        """
        raise NotImplementedError

    def infer(self, data):
        """
        Args:
            data <torch.Tensor>: (batch_size, num_samples, num_features)
        Returns:
            cluster_ids <torch.Tensor>: (batch_size, num_samples)
        """
        raise NotImplementedError

def init_centroids_random(data, K=2):
    """
    Args:
        data <torch.Tensor>: (batch_size, num_samples, num_features)
        K <int>: # of clusters
    Returns:
        centroid_ids <torch.LongTensor>: (batch_size, K)
    """
    batch_size, num_samples, _ = data.size()
    centroid_ids = []

    for _ in range(batch_size):
        _centroid_ids = torch.randperm(num_samples)[:K]
        centroid_ids.append(_centroid_ids)
    
    centroid_ids = torch.stack(centroid_ids, dim=0) # (batch_size, K)
    centroid_ids = centroid_ids.to(data.device)

    return centroid_ids

def init_kmeans_pp(data, K=2, compute_distance=lambda x, y, dim=-1: torch.norm(x - y, dim=dim)):
    """
    Args:
        data <torch.Tensor>: (batch_size, num_samples, num_features)
    Returns:
        centroid_ids <torch.LongTensor>: (batch_size, K)
    """
    _, num_samples, _ = data.size()

    centroid_ids = []

    for _data in data:
        _centroid_ids = torch.randperm(num_samples)[:1]
        _centroid_ids = _centroid_ids.to(_data.device)

        for _ in range(K - 1):
            centroids = _data[_centroid_ids] # (num_samples, num_features)

            distance = compute_distance(_data.unsqueeze(dim=1), centroids, dim=2) # (num_samples, K)
            distance, _ = torch.min(distance, dim=1)
            weights = distance / torch.sum(distance)
            _centroid_id = torch.multinomial(weights, 1) # equals to categorical distribution.
            _centroid_ids = torch.cat([_centroid_ids, _centroid_id], dim=0)
        
        centroid_ids.append(_centroid_ids)
    
    centroid_ids = torch.stack(centroid_ids, dim=0)

    return centroid_ids

def _test_kmeans_pp_iteration():
    K = 2
    iteration = 10
    seed = 111

    df = pd.read_csv("data/faithful.csv")
    x, y = torch.Tensor(df['waiting']).unsqueeze(dim=1), torch.Tensor(df['eruptions']).unsqueeze(dim=1)
    data0 = torch.cat([x, y], dim=1)

    mat = torch.Tensor([[1, -0.1], [-0.1, 0.8]])
    data1 = torch.matmul(data0, mat)

    data = torch.stack([data0, data1], dim=0)
    mean = data.mean(dim=1, keepdim=True)
    std = data.std(dim=1, keepdim=True)
    data = (data - mean) / std

    for batch_idx, _ in enumerate(data):
        os.makedirs("data/KMeans/iteration/{}".format(batch_idx + 1), exist_ok=True)

    for batch_idx, _data in enumerate(data):
        plt.figure()
        x, y = torch.unbind(_data, dim=-1)
        plt.scatter(x, y, color='black')
        plt.savefig("data/KMeans/iteration/{}/faithful-0.png".format(batch_idx + 1), bbox_inches='tight')
        plt.close()

    torch.manual_seed(seed)

    kmeans = KMeans(K=K)
    _, centroids = kmeans(data, iteration=iteration) # (batch_size, K), (batch_size, K, num_features)

    for batch_idx, (_data, _centroids) in enumerate(zip(data, centroids)):
        plt.figure()
        x, y = torch.unbind(_data, dim=-1)
        plt.scatter(x, y, color='black')
        x, y = torch.unbind(_centroids, dim=-1)
        plt.scatter(x, y, color='red')
        plt.savefig("data/KMeans/iteration/{}/faithful-last.png".format(batch_idx + 1), bbox_inches='tight')
        plt.close()

    # or same as ...
    torch.manual_seed(seed)

    kmeans = KMeans(K=K)
    _, centroids = kmeans(data, iteration=0) # Only initializes centroids.

    for idx in range(iteration):
        _, centroids = kmeans(data, iteration=1)

        for batch_idx, (_data, _centroids) in enumerate(zip(data, centroids)):
            plt.figure()
            x, y = torch.unbind(_data, dim=-1)
            plt.scatter(x, y, color='black')
            x, y = torch.unbind(_centroids, dim=-1)
            plt.scatter(x, y, color='red')
            plt.savefig("data/KMeans/iteration/{}/faithful-{}.png".format(batch_idx + 1, idx + 1), bbox_inches='tight')
            plt.close()

def _test_kmeans():
    K = 2
    seed = 111

    df = pd.read_csv("data/faithful.csv")
    x, y = torch.Tensor(df['waiting']).unsqueeze(dim=1), torch.Tensor(df['eruptions']).unsqueeze(dim=1)
    data0 = torch.cat([x, y], dim=1)

    mat = torch.Tensor([[1, -0.1], [-0.1, 0.8]])
    data1 = torch.matmul(data0, mat)

    data = torch.stack([data0, data1], dim=0)
    mean = data.mean(dim=1, keepdim=True)
    std = data.std(dim=1, keepdim=True)
    data = (data - mean) / std

    for batch_idx, _ in enumerate(data):
        os.makedirs("data/KMeans/None/{}".format(batch_idx + 1), exist_ok=True)

    for batch_idx, _data in enumerate(data):
        plt.figure()
        x, y = torch.unbind(_data, dim=-1)
        plt.scatter(x, y, color='black')
        plt.savefig("data/KMeans/None/{}/faithful-0.png".format(batch_idx + 1), bbox_inches='tight')
        plt.close()

    torch.manual_seed(seed)

    kmeans = KMeans(K=K)
    _, centroids = kmeans(data) # (batch_size, K), (batch_size, K, num_features)

    for batch_idx, (_data, _centroids) in enumerate(zip(data, centroids)):
        plt.figure()
        x, y = torch.unbind(_data, dim=-1)
        plt.scatter(x, y, color='black')
        x, y = torch.unbind(_centroids, dim=-1)
        plt.scatter(x, y, color='red')
        plt.savefig("data/KMeans/None/{}/faithful-last.png".format(batch_idx + 1), bbox_inches='tight')
        plt.close()

def _test_gmm_clustering():
    K = 2
    seed = 111

    df = pd.read_csv("data/faithful.csv")
    x, y = torch.Tensor(df['waiting']).unsqueeze(dim=1), torch.Tensor(df['eruptions']).unsqueeze(dim=1)
    data0 = torch.cat([x, y], dim=1)

    mat = torch.Tensor([[1, -0.1], [-0.1, 0.8]])
    data1 = torch.matmul(data0, mat)

    data = torch.stack([data0, data1], dim=0)
    mean = data.mean(dim=1, keepdim=True)
    std = data.std(dim=1, keepdim=True)
    data = (data - mean) / std

    for batch_idx, _ in enumerate(data):
        os.makedirs("data/GMMClustering/None/{}".format(batch_idx + 1), exist_ok=True)

    for batch_idx, _data in enumerate(data):
        plt.figure()
        x, y = torch.unbind(_data, dim=-1)
        plt.scatter(x, y, color='black')
        plt.savefig("data/GMMClustering/None/{}/faithful-0.png".format(batch_idx + 1), bbox_inches='tight')
        plt.close()

    torch.manual_seed(seed)

    gmm_clustering = GMMClustering(K=K)
    _, centroids = gmm_clustering(data) # (batch_size, K), (batch_size, K, num_features)

    for batch_idx, (_data, _centroids) in enumerate(zip(data, centroids)):
        plt.figure()
        x, y = torch.unbind(_data, dim=-1)
        plt.scatter(x, y, color='black')
        x, y = torch.unbind(_centroids, dim=-1)
        plt.scatter(x, y, color='red')
        plt.savefig("data/GMMClustering/None/{}/faithful-last.png".format(batch_idx + 1), bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    print("KMeans++, iteration")
    _test_kmeans_pp_iteration()
    print()

    print("KMeans")
    _test_kmeans()
    print()

    print("GMM clusteing")
    _test_gmm_clustering()