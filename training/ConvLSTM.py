# +
import torch
import torch.nn as nn
import lightning.pytorch as pl

from models.ConvLSTMCell import ConvLSTMCell

class ConvLSTM(pl.LightningModule):
    
    def __init__(self, in_channels, out_channels, kernel_size, padding, activation, frame_size):
        
        super(ConvLSTM, self).__init__()
        
        self.out_channels = out_channels
        
        # We'll unroll this over time steps
        self.convLSTMCell = ConvLSTMCell(in_channels, out_channels, kernel_size, padding, activation, frame_size)
        
    def forward(self, X):
        
        # X is a sequence (batch_size, seq_len, num_channnels, height, width)
        
        # Get the dimensions
        batch_size, seq_len, _, height, width = X.size()
        
        # Initialize output
        output = torch.zeros(batch_size, seq_len, self.out_channels, height, width)           
        output = output.to(X)
        
        # Initialize hidden state
        H = torch.zeros(batch_size, self.out_channels, height, width)
        H = H.to(X)
        
        # Initialize Cell input
        C = torch.zeros(batch_size, self.out_channels, height, width)         
        C = C.to(X)
        
        
        # Unroll over time steps
        for time_step in range(seq_len):
            H, C = self.convLSTMCell(X[:, time_step, :], H, C)
            output[:, time_step, :] = H
        
        # Will return all of the hidden states of the LSTM
        return output
