# +
import torch
import torch.nn as nn
import lightning.pytorch as pl

# Original ConvLSTM cell proposed by shi et al.

class ConvLSTMCell(pl.LightningModule):
    
    def __init__(self, in_channels, out_channels, kernel_size, padding, activation, frame_size):
        
        super(ConvLSTMCell, self).__init__()
        
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation =="relu":
            self.activation = torch.relu
        
        # Idea adapted from https://github.com/ndrpls/ConvLSTM_pytorch
        self.conv = nn.Conv2d(in_channels = in_channels + out_channels,
                             out_channels = 4 * out_channels,
                             kernel_size = kernel_size,
                             padding = padding)
        
        # Initialize the weights for Hadamard Products
        self.W_ci = nn.Parameter(torch.zeros(out_channels, *frame_size))
        self.W_co = nn.Parameter(torch.zeros(out_channels, *frame_size))
        self.W_cf = nn.Parameter(torch.zeros(out_channels, *frame_size))
    
    def forward(self, X, H_prev, C_prev):
        
        # Idea adapted from https://github.com/ndrpls/ConvLSTM_pytorch
        # previous dim=1 now dim=2
        #print(X.size())
        #print(H_prev.size())
        #print(torch.cat([X, H_prev], dim=2).size())
        
        conv_output = self.conv(torch.cat([X, H_prev], dim=1))
        
        i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)
        
        input_gate = torch.sigmoid(i_conv + self.W_ci * C_prev)
        forget_gate = torch.sigmoid(f_conv + self.W_cf * C_prev)
        
        # Current cell output
        C = forget_gate * C_prev + input_gate * self.activation(C_conv)
        
        output_gate = torch.sigmoid(o_conv + self.W_co * C)

        
        # Current hiddenstate
        H = output_gate * self.activation(C)
        
        return H, C
