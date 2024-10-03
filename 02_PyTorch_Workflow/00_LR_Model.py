import torch 
from torch import nn

# Create a Linear Regression model in PyTorch.
class LinearRegressionModule(nn.Module):
    '''Subclass -> nn.Module (this contains all the building bocks for neural networks)'''
    def __init__(self):
        super.__init__()

        '''Initialize model parameters to be used in various computations (these could be different layers from torch.nn, single parameters, hard-coded values or functions).'''
        # Initialize Model parameters. 
        self.weights = nn.Parameter(torch.randn(1,
                                                requires_grad=True, 
                                                dtype=torch.float))
        '''requires_grad = True means PyTorch will track the gradients of this specific parameter for use with torch.autograd (torch.autograd <- implements gradient descent) and gradient descent (for many torch.nn modules, requires_grad=True is set by default)'''
        self.bias = nn.Parameter(torch.randn(1, 
                                             requires_grad=True,
                                             dtype=torch.float))
    
    # forward() defines the computation in the model. 
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        '''Any subclass of nn.Module needs to overwrite forward() (this defines the forward computation of the model). In this case, forward() method implements the LinearRegressions Module.'''
        return self.weights * x + self.bias
    
# Create a random seed
torch.manual_seed(42)

# Create an instance of the model (this is a subclass of nn.Module)
model_0 = LinearRegressionModel()

# Check out the parameters.
list(model_0.parameters())

# List named parameters. 
model_0.state_dict()

# Make predictions with model 
with torch.inference_mode():
    y_prds = model_0(X_test)

y_preds