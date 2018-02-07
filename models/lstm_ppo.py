import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from models.helpers import to_numpy
from models.joint import JointPPO

# --

class LSTMSoftmaxPPO(JointPPO):
    
    emb_dim     = 64
    lstm_dim    = 128
    num_layers  = 1
    temperature = 100
    
    def __init__(self, n_inputs=32, n_outputs=4, output_channels=10,
        entropy_penalty=0.0, adam_lr=None, adam_eps=None, clip_eps=None, cuda=True):
        
        super(LSTMSoftmaxPPO, self).__init__()
        
        # n_inputs        -> state dimension
        # n_outputs       -> number of steps
        # output_channels -> cardinality of each choice
        
        self.n_outputs = n_outputs
        
        self.emb  = nn.Embedding(output_channels, self.emb_dim)
        self.lstm = nn.LSTM(self.emb_dim, self.lstm_dim, num_layers=self.num_layers)
        
        self.policy_fc = nn.Linear(self.lstm_dim, output_channels)
        self.value_fc = nn.Linear(self.lstm_dim, 1)
        
        if adam_lr and adam_eps:
            self.opt = torch.optim.Adam(self.parameters(), lr=adam_lr, eps=adam_eps)
            self.clip_eps = clip_eps
            self.entropy_penalty = entropy_penalty
            
            self._old = LSTMSoftmaxPPO(n_inputs, n_outputs, cuda=cuda)
        
        self._cuda = cuda
    
    def _lstm_forward(self, x, inner):
        e = self.emb(x)
        e = e.unsqueeze(0)
        lstm_out, inner = self.lstm(e, inner)
        lstm_out = lstm_out.squeeze(0)
        return lstm_out, inner
    
    def sample_actions(self, states):
        # !! The dimensions here are tricky
        
        inner = (
            Variable(torch.zeros(self.num_layers, states.shape[0], self.lstm_dim)),
            Variable(torch.zeros(self.num_layers, states.shape[0], self.lstm_dim)),
        )
        
        all_actions = [
            Variable(torch.LongTensor([0] * states.shape[0]))
        ]
        for i in range(self.n_outputs):
            
            # Run LSTM cell
            lstm_out, inner = self._lstm_forward(all_actions[-1], inner)
            
            # Compute policy
            policy = self.policy_fc(lstm_out)
            
            # Sample action
            action = F.softmax(policy * self.temperature, dim=-1)
            action = action.multinomial().squeeze()
            all_actions.append(action)
        
        # Compute value from final state (?)
        lstm_out, _ = self._lstm_forward(all_actions[-1], inner)
        value_predictions = self.value_fc(lstm_out)
        
        return to_numpy(torch.stack(all_actions, dim=-1)), to_numpy(value_predictions)
    
    def evaluate_actions(self, states, actions):
        # !! The dimensions here are tricky
        
        inner = (
            Variable(torch.zeros(self.num_layers, states.shape[0], self.lstm_dim)),
            Variable(torch.zeros(self.num_layers, states.shape[0], self.lstm_dim)),
        )
        
        all_action_log_probs = []
        dist_entropy = None
        for i in range(actions.shape[1]):
            
            current_actions = actions[:,i].contiguous()
            # Run LSTM cell
            lstm_out, inner = self._lstm_forward(current_actions, inner)
            
            # Compute policy
            policy = self.policy_fc(lstm_out)
            
            # Compute log probs
            log_probs = F.log_softmax(policy * self.temperature, dim=-1)
            action_log_probs = log_probs.gather(1, current_actions.view(-1, 1))
            all_action_log_probs.append(action_log_probs)
            
            # Compute entropy
            probs = F.softmax(policy * self.temperature, dim=-1)
            if dist_entropy is not None:
                dist_entropy += -(log_probs * probs)
            else:
                dist_entropy = -(log_probs * probs)
        
        # Compute value from final state (?)
        value_predictions = self.value_fc(lstm_out)
        
        return value_predictions, torch.cat(all_action_log_probs, dim=1), dist_entropy.mean()

# --

if __name__ == "__main__":
    lstm_ppo = LSTMSoftmaxPPO()
    
    states = torch.zeros(8).long()
    actions, value_predictions_1 = lstm_ppo.sample_actions(states)
    
    value_predictions_2, action_log_probs, dist_entropy = lstm_ppo.evaluate_actions(states, Variable(torch.LongTensor(actions)))
    
    assert (value_predictions_1 == value_predictions_2.data.numpy()).all()


