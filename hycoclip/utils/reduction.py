import os
import sys
import logging
import torch


class HoroPCAReduction:
    
    @staticmethod
    def import_packages():
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../HoroPCA/')))
        import geom.poincare as poincare
        from learning.frechet import Frechet
        from learning.pca import HoroPCA
        
        return poincare, Frechet, HoroPCA
    
    def __init__(self, dim, n_components=2, lr=5e-2, max_steps=500, downsample=10000):
        self.dim = dim
        self.n_components = n_components
        self.lr = lr
        self.max_steps = max_steps
        self.downsample = downsample
        
        self.manifold, self.frechet, HoroPCA = self.import_packages()
        self.model = HoroPCA(dim=dim, n_components=n_components, lr=lr, max_steps=max_steps)
        if torch.cuda.is_available():
            self.model.cuda()
            
    def _to_torch(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).to(torch.float32)
        if torch.cuda.is_available():
            x = x.cuda()
        return x
    
    def _to_numpy(self, x):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        return x
    
    def _center_data(self, x):
        logging.info("Computing the Frechet mean to center the embeddings")
        frechet = self.frechet(lr=1e-2, eps=1e-5, max_steps=5000)
        mu_ref, has_converged = frechet.mean(x, return_converged=True)
        logging.info(f"Mean computation has converged: {has_converged}")
        x = self.manifold.reflect_at_zero(x, mu_ref)
        return x

    def fit(self, x, center=False):
        x = self._to_torch(x)
        if self.downsample is not None and x.shape[0] > self.downsample:
            idxs = torch.randperm(x.shape[0])
            x = x[idxs[:self.downsample]]
        
        if center:
            # Compute the mean and center the data
            x = self._center_data(x)

        self.model.fit(x, iterative=False, optim=True)
        metrics = self.model.compute_metrics(x)

        return metrics

    def transform(self, x, center=False):
        x = self._to_torch(x)
        if center:
            x = self._center_data(x)
        x_proj = self._to_numpy(self.model.map_to_ball(x))
        return x_proj
    
    def fit_transform(self, x, center=False):
        self.fit(x, center=center)
        x_proj = self.transform(x, center=center)
        return x_proj


class COSNEReduction:
    
    @staticmethod
    def import_packages():
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../CO-SNE/')))
        import hyptorch.pmath as pmath
        from htsne_impl import TSNE as hTSNE
        
        return pmath, hTSNE

    def __init__(self, n_components, lr=5.0, lr_h_loss=0.1, max_steps=500, perplexity=20, early_exaggeration=1.0, student_t_gamma=0.1, downsample=None):

        self.n_components = n_components
        self.lr = lr
        self.lr_h_loss = lr_h_loss
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.student_t_gamma = student_t_gamma
        self.downsample = downsample
        
        self.manifold, hTSNE = self.import_packages()
        self.model = hTSNE(n_components=n_components, verbose=0, method='exact', square_distances=True,
                           metric='precomputed', learning_rate_for_h_loss=lr_h_loss, student_t_gamma=student_t_gamma,
                           learning_rate=lr, n_iter=max_steps, perplexity=perplexity, early_exaggeration=early_exaggeration)
        
    def _to_torch(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).to(torch.float32)
        if torch.cuda.is_available():
            x = x.cuda()
        return x
    
    def _to_numpy(self, x):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        return x

    def fit_transform(self, x, c=1.0):
        if self.downsample is not None and x.shape[0] > self.downsample:
            idxs = torch.randperm(x.shape[0])
            x = x[idxs[:self.downsample]]
        x_fit = self._to_torch(x)
        dists = self._to_numpy(self.manifold.dist_matrix(x_fit, x_fit, c=c))
        CO_SNE_embedding = self.model.fit_transform(dists, x_fit)

        return CO_SNE_embedding