import torch
from torch import nn
import numpy as np
import math

class Lighter(nn.Module):
  """_summary_
    Lighter model
  """
  
  def __init__(self, window_radius=2):
    super().__init__()
    self.window_radius = window_radius
    
    # attention block
    self.block_identification_layer = nn.Sequential(
        nn.Linear(9, 128, bias=False),       
        nn.ReLU(),
        nn.Linear(128, 32, bias=False),   
        nn.ReLU(),
        nn.Linear(32,16, bias=False),
        nn.ReLU(),
        nn.Linear(16,1, bias=False)
    )
    self.block_adjency_matrix = None
    self.cells_position = None
    self.attention_layer = nn.Sequential(
        nn.Linear((self.window_radius*2 + 1)**2, 128, bias=False),
        nn.ReLU(),
        nn.Linear(128, 64, bias=False),
        nn.ReLU(),
        nn.Linear(64,32, bias=False),
        nn.ReLU(),
        nn.Linear(32,16, bias=False),
        nn.ReLU(),
        nn.Linear(16,1, bias=False),
    )
    self.attention_softmax = nn.Softmax(dim=1) 
    self.attention_relu = nn.ReLU()
    self.relu_attention_bias = nn.Parameter(torch.Tensor([1e-16]), requires_grad=True)
    self.classification_layer =  nn.Sequential(
      nn.Linear(900, 1024, bias=False),
      nn.ReLU(),
      nn.Linear(1024, 512, bias=False),
      nn.ReLU(),
      nn.Linear(512, 128, bias=False),
      nn.ReLU(),
      nn.Linear(128, 64, bias=False),
      nn.ReLU(),
      nn.Linear(64, 10, bias=False),
    )


  def forward(self, X: torch.Tensor, A: torch.Tensor):
    ## Parameters description:
    # X: features column vector;
    # A: adjency matrix

    # blocks identification
    image_size = int(math.sqrt(X.shape[1]))
    X = X * 255 + 100
    B = X.reshape(-1, image_size, image_size)
    B = B.unfold(1, 3, 3).unfold(2,3,3).reshape(-1, 100, 9)
    B = self.block_identification_layer(B).squeeze(2)
    max_values, _ = torch.max(B, dim=1, keepdim=True)
    min_values, _ = torch.min(B, dim=1, keepdim=True)
    B = ((B - min_values) / (max_values - min_values)) * 255 + 100
    
    # blocks attention evaluation
    if self.block_adjency_matrix == None:
      self.block_adjency_matrix, self.cells_position = self.attention_adjm(n_nodes=B.shape[1], window_radius=self.window_radius)
      self.block_adjency_matrix = self.block_adjency_matrix.unsqueeze(0)
      self.block_adjency_matrix = self.block_adjency_matrix.to(X.device)
      self.cells_position = self.cells_position.to(X.device)
      
    B = B.unsqueeze(1).to(X.device)
    BA = B * self.block_adjency_matrix
    BA = BA[BA != 0]
    BA = BA.float()
    B = B.squeeze(1)
    
    N = torch.stack([self.cells_position] * X.shape[0])
    N = N.masked_scatter(self.cells_position == 1, BA)
    
    E = self.attention_layer(N).squeeze(2)
    max_values, _ = torch.max(E, dim=1, keepdim=True)
    min_values, _ = torch.min(E, dim=1, keepdim=True)
    E = ((E - min_values) / (max_values - min_values)).reshape((-1, int(math.sqrt(B.shape[1])), int(math.sqrt(B.shape[1]))))
    E = E - (self.relu_attention_bias**2)
    E = self.attention_relu(E)
    E = torch.repeat_interleave(E, 3, dim=1)
    E = torch.repeat_interleave(E, 3, dim=2).flatten(1,2)
    X = X * E
    
    # # classification
    output = self.classification_layer(X)
    
    # # saving important parameters
    self.X = X
    self.B = B.detach()
    self.E = E.detach()
    
    return output

  def attention_adjm(self, n_nodes, window_radius, stack = 1):
    matrix_side = int(np.sqrt(n_nodes))
    A = torch.zeros((n_nodes, n_nodes))
    cell_presence = torch.full((n_nodes, (2*window_radius + 1)**2), -1)

    for node in range(0, n_nodes): 
      row, col = node // matrix_side, node % matrix_side
      cell = 0
      for i in range(-window_radius, window_radius + 1):
        for j in range(-window_radius, window_radius + 1):
          new_node = (row + i)*matrix_side + (col + j)
          if 0 <= (row + i) < matrix_side and 0 <= (col + j) < matrix_side:
            A[node, new_node] = 1
            cell_presence[node, cell] = 1
          cell += 1
    
    return A, cell_presence.float()