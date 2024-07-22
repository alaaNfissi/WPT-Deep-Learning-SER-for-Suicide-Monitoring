#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Title: Speech Emotion Recognition Model Definition
Author: Alaa Nfissi
Date: July 21, 2024
Description: Defines the architecture of the speech emotion recognition model, incorporating 
custom layers, attention mechanisms, and the overall neural network structure.
"""


from custom_layers import *

class CNN1DSABiGRUTA(nn.Module):
    
    """
    This class implements a Convolutional Neural Network (CNN) with 1D dilated convolutions,
    Spatial Attention, Bidirectional Gated Recurrent Units (Bi-GRU), and Temporal Attention layers.
    
    Parameters:
    - n_input: The number of input channels.
    - n_channel: The number of output channels for the convolutional layers.
    - hidden_dim: The hidden dimension size for the GRU layers.
    - n_layers: The number of layers in the GRU.
    - normalize_attn: Boolean indicating whether to normalize attention weights.
    """
    
    def __init__(self, n_input, n_channel, hidden_dim, n_layers, normalize_attn=True):
        super().__init__()
        self.n_channel = n_channel
        self.n_input = n_input
        self.normalize_attn = normalize_attn
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        
        ################################################ 1D CNN ################################################################
        
        # Define the first convolutional layer with dilation
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=7, stride=4, dilation=3)
        self.in1 = nn.InstanceNorm1d(n_channel)
        self.relu1 = nn.LeakyReLU()

        self.dropout1 = nn.Dropout(0.2)
        
        # Define the second convolutional layer with dilation
        self.conv1_1 = nn.Conv1d(n_channel, 2*n_channel, kernel_size=5, stride=4, dilation=2)
        self.in1_1 = nn.InstanceNorm1d(2*n_channel)
        self.relu1_1 = nn.LeakyReLU()

        self.dropout2 = nn.Dropout(0.2)
        
        ############################################ Spatial Attention #########################################################
        
        # Define spatial attention components
        self.dense1 = nn.Conv1d(in_channels=2*n_channel, out_channels=2*n_channel, kernel_size=7, padding=3, bias=True)
        self.in1_2 = nn.InstanceNorm1d(2*n_channel)
        self.relu1_2 = nn.LeakyReLU()
        
        # Define the spatial attention layer
        self.spatialAttn= SpatialAttentionBlock(in_features=2*n_channel, normalize_attn=self.normalize_attn)
        
        self.reluAtt1 = nn.LeakyReLU()
        
        
        #################################################### Bi-GRU #############################################################
        
        # Define the Bi-GRU layer
        self.gru1 = nn.GRU(2*n_channel, self.hidden_dim, self.n_layers, batch_first=True, bidirectional=True, dropout=0)
        
        
        ############################################ Temporal Attention #########################################################
        
        # Define the temporal attention layer
        self.tempAttn = TemporalAttn(hidden_size=2*hidden_dim)
        
        
    def forward(self, x, h):
        
        """
        Forward pass of the CNN1DSABiGRUTA model.
        
        Parameters:
        - x: The input tensor.
        - h: The initial hidden state for the GRU layer.
        
        Returns:
        - The output tensor after passing through the CNN, spatial attention, Bi-GRU, and temporal attention layers.
        - The updated hidden state.
        """
        
        x = self.conv1(x)
        x = self.relu1(self.in1(x))
        x = self.dropout1(x)
        
        x = self.conv1_1(x)
        x = self.relu1_1(self.in1_1(x))
        x = self.dropout2(x)

        g1 = self.dense1(x)
        g1 = self.relu1_2(self.in1_2(g1))
        
        c, g_attended_1 = self.spatialAttn(x, g1)
        
        x = self.reluAtt1(x + g_attended_1.unsqueeze(2))  # Residual connection with attended global features
        
        x = x.permute(0, 2, 1)
        
        x, h = self.gru1(x, h)
        
        x, weights = self.tempAttn(x)
        
        x = x.unsqueeze(1)
        
        return x, h
    
class ChannelWeighting(nn.Module):
    
    """
    This layer applies a learnable weighting to the channels of its input tensor.
    
    Parameters:
    - num_channels: The number of channels in the input tensor.
    """
    
    def __init__(self, num_channels):
        super(ChannelWeighting, self).__init__()
        
        # Initialize weights for each channel. These weights are learnable parameters.
        self.weights = nn.Parameter(torch.ones(num_channels))
        
    def forward(self, x):
        
        """
        Forward pass of the ChannelWeighting layer.
        
        Parameters:
        - x: Input tensor of shape (batch_size, num_channels, length).
        
        Returns:
        - The input tensor with each channel weighted by the learned weights.
        """
        
        # Apply weights to each channel. The weights are broadcasted to match the input tensor shape.
        return x * self.weights.view(1, -1, 1)


class WavRecDec(nn.Module):

    """ Wavelet Recursive Decomposition
    # This class implements the recursive decomposition of the wavelet transform.
    # The decomposition is performed using the wavelet packet transform and hard thresholding.
    # The wavelet packet transform is implemented using a series of convolutional layers.
    # The hard thresholding is implemented using the HardThresholdAssym layer.
    # The decomposition is performed recursively until the desired level is reached.
    # The wavelet transform is performed in the forward pass.
    # 
    # Parameters:
    # - level: Number of decomposition levels.
    # - kernelsG: List of convolutional kernels for the low-pass filters.
    # - kernelsH: List of convolutional kernels for the high-pass filters.
    #   The kernels are initialized based on the constraint specified.
    # - LowPassWave: List of low-pass wavelet transform layers.
    # - HighPassWave: List of high-pass wavelet transform layers.
    # - HardThresholdAssymH: List of asymmetric hard thresholding layers.
    # - currentLevel: Current decomposition level.
    # - levelTrack: List to track the current level.
    #   The list is used to keep track of the current level during the recursive decomposition.
    #   The list is updated in each forward pass.
    # 
    # Returns:
    # - The wavelet coefficients after decomposition.
    #   The coefficients are concatenated along the channel dimension."""


    def __init__(self, level, kernelsG, kernelsH, LowPassWave, HighPassWave, HardThresholdAssymH, currentLevel, levelTrack):
        super().__init__()
        self.level = level
        self.kernelsG = kernelsG
        self.kernelsH = kernelsH
        self.LowPassWave = LowPassWave
        self.HighPassWave = HighPassWave
        self.HardThresholdAssymH = HardThresholdAssymH
        self.currentLevel = currentLevel
        self.levelTrack = levelTrack

        
    def forward(self, x):
        if self.currentLevel < self.level:
            self.levelTrack.append(self.currentLevel)
            idx = self.levelTrack.count(self.currentLevel) - 1
            
            h = self.HighPassWave[self.currentLevel][idx]([x, self.kernelsH[self.currentLevel][idx]])
            h = self.HardThresholdAssymH[self.currentLevel][idx](h)
        
            g = self.LowPassWave[self.currentLevel][idx]([x, self.kernelsG[self.currentLevel][idx]])
            
            self.currentLevel += 1
            
            wrd_h = WavRecDec(self.level, self.kernelsG, self.kernelsH, self.LowPassWave, self.HighPassWave, self.HardThresholdAssymH, self.currentLevel, self.levelTrack)(h)
            wrd_g = WavRecDec(self.level, self.kernelsG, self.kernelsH, self.LowPassWave, self.HighPassWave, self.HardThresholdAssymH, self.currentLevel, self.levelTrack)(g)
            
            wrd_h_g = torch.cat((wrd_h, wrd_g), 1)

            return wrd_h_g
        return x

class WPTDL(nn.Module):
    
    """
    The model combines a Learnable Fast Discrete Wavelet Transform (LFDWT) with a 1D CNN,
    spatial attention, Bi-GRU, and temporal attention layers for speech emotion recognition (SER).
    
    Parameters:
    - n_input: Number of input channels.
    - hidden_dim: Dimension of hidden layers in GRU.
    - n_layers: Number of layers in the GRU.
    - n_output: Number of output classes.
    - stride: Stride size for convolution operations.
    - n_channel: Number of channels in convolution layers.
    - inputSize: Size of the input signal.
    - kernelInit: Initial value or method for kernel initialization.
    - kernTrainable: Indicates if kernels are trainable.
    - level: Number of decomposition levels in wavelet transform.
    - kernelsConstraint: Type of constraint on kernels ('CQF', 'PerLayer', or 'PerFilter').
    - initHT: Initial value for hard thresholding.
    - trainHT: Indicates if hard thresholding is trainable.
    - alpha: Alpha value for LAHT.
    """
    
    def __init__(self, n_input, hidden_dim, n_layers, n_output, stride=2, n_channel=128, 
                 inputSize=None, kernelInit=20, kernTrainable=True, level=1, kernelsConstraint='QMF', initHT=1.0, trainHT=True, alpha=10):
        super().__init__()
        self.n_input = n_input
        self.n_channel = n_channel
        self.n_output = n_output
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.inputSize = inputSize
        self.kernelInit = kernelInit
        self.kernTrainable = kernTrainable
        self.level = level
        self.kernelsConstraint = kernelsConstraint
        self.initHT = initHT
        self.trainHT = trainHT
        self.alpha = alpha
        
        # Initialization of kernels based on the constraint specified
        if self.kernelsConstraint=='CQF':
            kr = Kernel(self.kernelInit, trainKern=self.kernTrainable)
            self.kernelsG_ = nn.ModuleList([nn.ModuleList([kr for l in range(2**lev)]) for lev in range(self.level)])
            self.kernelsG  = [[kern(None) for kern in self.kernelsG_[lev]] for lev in range(self.level)]
            self.kernelsH  = self.kernelsG

        elif self.kernelsConstraint=='PerLayer':
            kr = [Kernel(self.kernelInit, trainKern=self.kernTrainable) for lev in range(self.level)]
            self.kernelsG_ = nn.ModuleList([nn.ModuleList([kr[lev] for l in range(2**lev)]) for lev in range(self.level)])
            self.kernelsG  = [[kern(None) for kern in self.kernelsG_[lev]] for lev in range(self.level)]
            self.kernelsH  = self.kernelsG
            
        elif self.kernelsConstraint=='PerFilter':
            self.kernelsG_  = nn.ModuleList([nn.ModuleList([Kernel(self.kernelInit, trainKern=self.kernTrainable) for l in range(2**lev)]) for lev in range(self.level)])
            self.kernelsG  = [[kern(None) for kern in self.kernelsG_[lev]] for lev in range(self.level)]
            self.kernelsH_  = nn.ModuleList([nn.ModuleList([Kernel(self.kernelInit, trainKern=self.kernTrainable) for l in range(2**lev)]) for lev in range(self.level)])
            self.kernelsH  = [[kern(None) for kern in self.kernelsH_[lev]] for lev in range(self.level)]
        

        # Wavelet transform layers
        self.LowPassWave = nn.ModuleList([nn.ModuleList([LowPassWave() for l in range(2**lev)]) for lev in range(self.level)])
        self.HighPassWave = nn.ModuleList([nn.ModuleList([HighPassWave() for l in range(2**lev)]) for lev in range(self.level)])
        
        self.HardThresholdAssymH = nn.ModuleList([nn.ModuleList([HardThresholdAssym(init=self.initHT,trainBias=self.trainHT, alpha=self.alpha) for l in range(2**lev)]) for lev in range(self.level)])
        
        #####################################################################################################   
        
        # CNN, spatial attention, and Bi-GRU layers for each wavelet transform level and the low-frequency component
        self.conv1ds = nn.ModuleList([CNN1DSABiGRUTA(self.n_input, self.n_channel, self.hidden_dim, self.n_layers, True) for i in range(2**self.level)])

        
        #####################################################################################################
        
        # Channel weighting to combine the wavelet transform levels and the low-frequency component
        self.channel_weighting = ChannelWeighting(2**self.level)
        
        # Final convolutional layer to produce the output
        self.conv1_3 = nn.Conv1d(2**self.level, self.n_output, kernel_size=5, stride=1, dilation=1)
        self.in1_3 = nn.InstanceNorm1d(self.n_output)
        self.relu1_3 = nn.LeakyReLU()
        
        
    def forward(self, x, h):
        
        """
        Forward pass through the model.
        
        Parameters:
        - x: Input tensor of shape (batch_size, n_input, sequence_length).
        - h: Initial hidden states for the GRU layers.
        
        Returns:
        - The output tensor of shape (batch_size, n_output).
        - The updated hidden states.
        """

        # Decomposition using wavelet transform and processing with CNNs, spatial attention, and Bi-GRU
        self.wavdecrec = WavRecDec(self.level, self.kernelsG, self.kernelsH, self.LowPassWave, self.HighPassWave, self.HardThresholdAssymH, self.currentLevel, self.levelTrack)

        x = self.wavdecrec(x)
        
        x_aux = torch.Tensor().to(device)
        for lev in range(2**self.level):
            x_l, h[lev] = self.conv1ds[lev](x[:, lev, :].unsqueeze(1), h[lev])
            x_aux = torch.cat((x_aux, x_l), 1)
        
        ################## Channel weighting ##################
        
        
        # Applying channel weighting
        x = self.channel_weighting(x_aux)
        
        ########################################################
        
        # Final processing to produce the output
        self.currentLevel = 0
        self.levelTrack = []
        
        x = self.conv1_3(x)
        x = self.relu1_3(self.in1_3(x))
        
        # Global average pooling
        x = x.mean(2)
        
        output = F.log_softmax(x, dim=1)
        return output , h
    
    def init_hidden(self, batch_size):
        
        """
        Initializes hidden states for each GRU layer.
        
        Parameters:
        - batch_size: The batch size.
        
        Returns:
        - A list of initial hidden states for each Bi-GRU layer.
        """
        
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers*2, batch_size, self.hidden_dim).zero_().to(device)
        return [hidden for i in range(2**self.level)]