import torch
import numpy as np

class ToGraph():
  """_summary_
    Used to convert an image into a graph structure.
  """

  def __init__(self):
    self.adjency_matrix = None
    self.all_saved = False
  
  def starndard_adjency_matrix_creation(self, X):
    _, height, width = X.shape
    nodes = height * width
    adjency_matrix = np.zeros((nodes, nodes))
    valid_neighbour = lambda row, col: row >= 0 and col >= 0 and row < height and col < width 
    
    for node in range(0, nodes):
      neighbours = []
      current_row = node // width
      current_col = node % width
      for i in range(-1,2):
        for j in range(-1,2):
          row = current_row + i
          col = current_col + j
          if valid_neighbour(row, col):
            neighbours.append(row * width + col)
            
      neighbours.remove(node)

      for neighbour in neighbours:
        adjency_matrix[neighbour][node] = 1
        adjency_matrix[node][neighbour] = 1
    
    return torch.from_numpy(adjency_matrix)

  def __call__(self, image):
    image = torch.Tensor(image).float()
    if not self.all_saved:
      self.adjency_matrix = self.starndard_adjency_matrix_creation(image)
      self.all_saved = True
    return (image.flatten(), self.adjency_matrix)
  
class NoisyImage():
  """_summary_
    Used to add some noise to an image.
  """
  def __init__(self):
    pass
    
  def __call__(self, image):
    noise = (torch.rand(image.shape)*0.1)
    noisy_image = torch.clone(image) + noise
    noisy_image[noisy_image > 1] = 1
    return noisy_image
