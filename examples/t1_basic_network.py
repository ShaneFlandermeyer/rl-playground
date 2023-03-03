import torch
import torch.nn as nn


class Network(nn.Module):
  """A basic fully-connected classifier network as a PyTorch module."""

  def __init__(self,
               n_inputs: int,
               n_classes: int,
               dropout_prob: float = 0.3,
               ) -> None:
    """
    The class constructor for the network.

    Parameters
    ----------
    n_inputs : int
        Number of input features
    n_classes : int
        Number of classes that must be classified
    dropout_prob : float, optional
        Dropout probability, by default 0.3.
        Dropout is a very common strategy to prevent overfitting. When added to a network, it randomly "removes" neurons with some probability. This encourages the network to learn a more robust representation of the data.
    """
    # Call the constructor for the nn.Module class, which sets up the basics of the network. This is required!
    super().__init__()
    
    # The Sequential class takes a list of layers and applies them in the order they were given, one after the other.
    self.layers = nn.Sequential(
        nn.Linear(n_inputs, 5),
        nn.ReLU(),
        nn.Linear(5, 20),
        nn.ReLU(),
        nn.Linear(20, n_classes),
        nn.Dropout(p=dropout_prob),
        nn.Softmax(dim=1),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Perform one forward pass through the network. That is, apply the network to the input data to get an output.

    Parameters
    ----------
    x : torch.Tensor
        Input data tensor

    Returns
    -------
    torch.Tensor
        Output data. In this network, the output data is a tensor of probabilities for each class.
    """
    return self.layers(x)
  
if __name__ == '__main__':
  # This code is executed any time this file is run directly, but not if it is imported from another file.
  net = Network(n_inputs=2, n_classes=3)
  input_tensor = torch.Tensor([[2, 3]])
  # Pass the data through the network (forward pass)
  out1 = net(input_tensor)
  # Can also call the forward method directly to get the same result
  out2 = net.forward(input_tensor) 
  print(net)
  print(out1)
  print(out2)
