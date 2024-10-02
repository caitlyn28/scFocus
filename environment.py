import random
import collections
import tqdm
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale, StandardScaler
from scipy.stats import multivariate_normal, gaussian_kde


def get_state(f1_, f2_, batch_, bins, n_genes, n_states):
    """  
    Compute various histograms based on input features and return a concatenated state vector.  
  
    Parameters  
    ----------  
    f1_ : array_like  
        First feature set.  
    f2_ : array_like  
        Second feature set, assumed to be a 2D array with at least two columns.  
    batch_ : array_like  
        Labels or outcomes corresponding to the feature sets.  
    bins : int, optional  
        Number of bins to use for histogram calculations.  
  
    Returns  
    -------  
    state : ndarray  
        Concatenated state vector containing normalized histogram counts.  
    auc : float  
        Area under the ROC curve score for the test set.  
  
    Raises  
    ------  
    ValueError  
        If the lengths of `f1_`, `f2_`, and `batch_` are not equal.  
        If `f2_` does not have at least two columns.  
  
    Notes  
    -----  
    This function splits the input data into training and test sets, fits a RandomForestClassifier  
    on the training data, and computes histograms based on the classifier's predicted  
    probabilities and the second feature set. The histograms are then normalized and  
    concatenated into a state vector. Additionally, the function computes the area under  
    the ROC curve for the test set.  
    """  
    gidx = np.argsort(f1_.mean(axis=1))[:n_genes]
    f1_tr, f1_te, y_tr, y_te = train_test_split(f1_[:,gidx], batch_)
    f1_tr = StandardScaler().fit_transform(f1_tr)
    classifier = LogisticRegression().fit(f1_tr, y_tr)
    prob_tr = classifier.predict_proba(f1_tr)
    prob_te = classifier.predict_proba(f1_te)
    f1_tr_bins = minmax_scale(np.histogram(prob_tr, bins=bins)[0])
    f1_te_bins = minmax_scale(np.histogram(prob_te, bins=bins)[0])
    f2_ls = []
    for i in range(n_states):
        f2_bins = minmax_scale(np.histogram(f2_[:,i], bins=bins)[0])
        f2_ls.append(f2_bins)
    f2_bins = np.hstack(f2_ls)
    state = np.hstack([f1_tr_bins,
                       f1_te_bins,
                       f2_bins,
                       f2_[:,:n_states].mean(axis=0),
                       f2_[:,:n_states].std(axis=0)]
                     )
    auc = roc_auc_score(y_te, prob_te[:,1])
    return state[np.newaxis,:], auc, sum([f2_[:,i].std() for i in range(n_states)]) / n_states



class Env:
    """  
    Class representing an environment for interacting with a dataset and simulating actions.  

    Parameters  
    ----------  
    f1 : ndarray  
        First set of features.  
    f2 : ndarray  
        Second set of features.  
    batch : ndarray  
        Array indicating batch or class labels for the samples.  
    max_steps : int, optional  
        Maximum number of steps allowed in an episode. Defaults to 50.
    n_sample : int
        Number of samples for the subset of data.
    bins : int
        Number of bins to use for histogram calculations.
    sigma : float
        Standard error for the multi variable normal distribution
        
    Attributes  
    ----------  
    f1 : ndarray  
        Stored first set of features.  
    f2 : ndarray  
        Stored second set of features.  
    batch : ndarray  
        Stored batch or class labels.  
    cnt : int  
        Counter for the number of steps taken in the current episode.  
    max_steps : int  
        Maximum number of steps allowed in an episode.  
    n_sample : int
        Number of samples for the subset of data.
    bins : int
        Number of bins to use for histogram calculations.
    sigma : float
        Standard error for the multi variable normal distribution
        
    Methods  
    -------  
    reset()  
        Resets the environment to an initial state and returns the state.  
    step(action)  
        Performs a step in the environment given an action.  
        Returns the next state, reward, and a flag indicating if the episode is done.  
    """
    def __init__(self, f1, f2, batch, max_steps, n_sample, bins, n_genes, n_states, sigma, auc_scale):
        self.f1            = f1
        self.f2            = f2
        self.batch         = batch
        self.cnt           = 0
        self.max_steps     = max_steps
        self.n_sample      = n_sample
        self.bins          = bins
        self.n_genes       = n_genes
        self.n_states      = n_states
        self.sigma         = sigma
        self.auc_scale     = auc_scale
        
    def reset(self):
        self.cnt         = 0
        idx_n            = pd.Series(np.where(self.batch == np.unique(self.batch)[0])[0]).sample(self.n_sample).to_list()
        idx_p            = pd.Series(np.where(self.batch == np.unique(self.batch)[1])[0]).sample(self.n_sample).to_list()
        f1_, f2_, batch_ = self.f1[idx_n+idx_p,:], self.f2[idx_n+idx_p,:], self.batch[idx_n+idx_p]
        state, _, _      = get_state(f1_, f2_, batch_, self.bins, self.n_genes, self.n_states)
        return state
    
    def step(self, action):
        self.cnt += 1
        action = action.ravel()
        mu = action[:int(action.shape[-1]/2)]
        logstd = action[int(action.shape[-1]/2):]
        std = np.log1p(np.exp(logstd))
        mn = multivariate_normal(mu, np.diag(self.sigma / (1 + np.exp(-std))))
        weights = minmax_scale(mn.logpdf(self.f2[:,:int(action.shape[-1]/2)]))
        mask_n = self.batch == np.unique(self.batch)[0]
        mask_p = self.batch == np.unique(self.batch)[1]
        idx_n = pd.Series(np.where(mask_n)[0]).sample(self.n_sample, weights=weights[mask_n]).to_list()
        idx_p = pd.Series(np.where(mask_p)[0]).sample(self.n_sample, weights=weights[mask_p]).to_list()
        f1_, f2_, batch_ = self.f1[idx_n+idx_p,:], self.f2[idx_n+idx_p,:], self.batch[idx_n+idx_p]
        next_state, auc, err = get_state(f1_, f2_, batch_, self.bins, self.n_genes, self.n_states)
        reward = np.array([auc * self.auc_scale - err])[np.newaxis,:]
        done = np.array([True] if self.cnt >= self.max_steps else [False])[np.newaxis,:]
        return next_state, reward, done

class ReplayBuffer:
    """  
    A replay buffer to store transitions for experience replay.  
    
    Attributes  
    ----------  
    buffer : collections.deque  
        A deque (double-ended queue) that stores the transitions with a maximum length set by capacity.  
    
    Methods  
    -------  
    add(state, action, reward, next_state, done)  
        Adds a transition to the replay buffer.  
    sample(batch_size)  
        Randomly samples a batch of transitions from the buffer and returns them in a numpy array format.  
    size()  
        Returns the current size of the replay buffer.
    """
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.vstack(state), np.vstack(action), np.vstack(reward), np.vstack(next_state), np.vstack(done)
    
    def size(self):
        return len(self.buffer)
    
def train_off_policy(env, agent, replay_buffer, num_episodes, minimal_size, batch_size):
    """  
    Train an agent using an off-policy reinforcement learning algorithm.  
  
    This function trains the agent by interacting with the environment, collecting  
    experiences, and storing them in a replay buffer. Once the replay buffer reaches  
    a minimal size, the agent is updated using a batch of samples from the buffer.  
  
    Parameters  
    ----------  
    env : Env  
        The environment in which the agent interacts.  
    agent : Agent  
        The reinforcement learning agent to be trained.  
    replay_buffer : ReplayBuffer  
        A replay buffer object to store and sample experiences.  
    num_episodes : int  
        The total number of episodes to train the agent.  
    minimal_size : int  
        The minimum number of experiences required in the replay buffer before  
        starting agent updates.  
    batch_size : int  
        The batch size used for sampling experiences from the replay buffer during  
        agent updates.  
  
    Returns  
    -------  
    return_list : list  
        A list containing the returns (cumulative rewards) obtained in each episode.  
  
    """  
    return_list = []
    for i in range(10):
        with tqdm.tqdm(total=int(num_episodes/10), desc='Iteration %d'%(i+1)) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {'states':b_s, 'actions':b_a, 'next_states':b_ns, 'rewards':b_r,'dones':b_d}
                    agent.update(transition_dict)
                
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode':'%d' % (num_episodes/10*i+i_episode+1),
                                     'return':'%.3f'%np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list