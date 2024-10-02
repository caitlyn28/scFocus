from .model import SAC
from .environment import Env, ReplayBuffer, train_off_policy
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.preprocessing import minmax_scale
import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class focus:
    def __init__(self, f1, f2, batch, hidden_dim=128, action_dim=4, max_steps=5, n_sample=100, bins=15, n_genes=500, n_states=5, auc_scale=5, capacity=10000, actor_lr=1e-4, critic_lr=1e-3, alpha_lr=1e-4, target_entropy=-1, tau=.005, gamma=.99, num_episodes=500, minimal_size=1000, batch_size=64, device=device):
        """
        Initialize the focus.  
  
        Parameters  
        ----------
        f1: array like
            Gene expression space of the original data
        f2: array like
            Latent space of the orignal data (usually PCA space)
        batch:
            Batch information of the original data
       
        """
        sigma             = max([f2[:,i].std() for i in range(int(action_dim / 2))])
        self.env          = Env(f1, f2, batch, max_steps, n_sample, bins, n_genes, n_states, sigma, auc_scale)
        self.memory       = ReplayBuffer(capacity)
        state_dim         = bins*(n_states+2) + n_states*2
        action_space      = (f2[:,:n_states].min().item(),f2[:,:n_states].max().item())
        self.model        = SAC(state_dim, hidden_dim, action_dim, action_space, actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma, device)
        self.num_episodes = num_episodes
        self.minimal_size = minimal_size
        self.batch_size   = batch_size
        
    def fit(self):
        return_list = train_off_policy(self.env, self.model, self.memory, self.num_episodes, self.minimal_size, self.batch_size)
        self.return_list = return_list
        return self
        
    def sample(self, steps):
        ls_weights = []
        state = self.env.reset()
        for i in range(steps):
            with torch.no_grad():
                action, _ = self.model.actor(torch.FloatTensor(state).to(device))
                action = action.cpu().numpy()
            action = action.ravel()
            mu = action[:int(action.shape[-1]/2)]
            sigma = action[int(action.shape[-1]/2):]
            sigma = np.clip(sigma, -10, 10)
            mn = multivariate_normal(mu, np.diag(self.env.sigma / (1 + np.exp(-sigma))))
            weights = minmax_scale(mn.logpdf(self.env.f2[:,:int(action.shape[-1]/2)]))
            ls_weights.append(weights)
            state, _, _ = self.env.step(action)
        weights = np.array(ls_weights).T
        return weights        
        