import torch                        # Model creation
import numpy as np                  # Mathematics (e.g. data creation)
import matplotlib.pyplot as plt     # Plotting 


""" Linear Regression Model
Class used to create a model, that will later on be used to predict the values of "w" and "b".
It's inheriting torch.nn.Module - a standard procedure of creating PyTorch models
"""
class LinearRegressionModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        """Constructor method
        
        Using inheritance of torch.nn.Module, LinearRegressionModel is created. This is the standard method for creating various other models too.  
        
        Args:
            input_size  (int): the number of input values
            output_size (int): the number of output values
        """
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size) # Applying standard linear transformation (f(x) = wx + b) 

    def forward(self, X):
        """Forward pass 
    
        Based on current value of "w", calculate the prediction for each value of "X".
        This is basically the point of the model - predicting the right "w".
        
        Args:
            X (torch.Tensor): The values used for making prediction based on current state of "w" and "b"
        """
        out = self.linear(X) # Function is a simple linear equation: f(x) = w * x + b
        return out


if __name__ == "__main__":
    # Use GPU, if possible; if not, use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create data 
    # Similarly to "lin_reg.py" and "lin_reg_2_variables.py", we create data by creating a column vector of values [-5, -4.6, -4.2, ..., 4.2, 4.6], and the appropriate "labels" (values of linear function f(x))
    x_data = np.arange(-5, 5, 0.4, dtype=np.float32).reshape(-1, 1) # Create a column vector of values
    y_data = 3 * x_data + 2 # The function the model is using to predict w and b
    # y_data = y_data + 0.4 * np.random.rand(x_data.shape[0], x_data.shape[1]) # Optional: Add Gaussian noise
    X = torch.tensor(x_data, dtype=torch.float32, requires_grad=True).to(device) # Create input tensor out of numpy column vector 
    Y = torch.tensor(y_data, dtype=torch.float32, requires_grad=True).to(device) # Create ideal output tensor based on function f(x)

    # Plot data
    plt.plot(x_data, y_data, "r+") # Plot point at (X[i], f(X[i]))
    plt.xlabel("x")                # Set the label for x-axis
    plt.ylabel("y")                # Set the label for y-axis
    plt.grid()                     # Show the grid, for ease of comprehension
    plt.show()                     # Show the plot!

    # Create the model
    input_size  = 1     # Just one value of X        
    output_size = 1     # Just one value of Y_prediction
    learning_rate = 0.1 # Also called "step", represents how quick we are descending

    model = LinearRegressionModel(input_size, output_size) # Create the model
    model.to(device)                                       # Port the model to desired device (GPU, ideally)

    criterion = torch.nn.MSELoss() # Criterion is "loss function", and in this case is "Mean Squared Error" function

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # Stochastic Gradient Descent optimizer is used to make steps that will impact values of w and b 

    # Start the training process
    epochs = 30 # Train model for "30" epochs (also called "iterations")

    for epoch in range(epochs):
        optimizer.zero_grad() # Zero out the gradient
        
        # Forward pass
        Y_predictions = model(X) # Forward pass 
        
        # Calculate the loss
        loss = criterion(Y_predictions, Y) # Calculate the loss based on predictions

        # Backward pass
        loss.backward()  # Calculate the gradient w.r.t "w" and "b"       
        optimizer.step() # Make a step opposite of gradient to make error smaller

        # Find out current values of w and b from model        
        w, b = -1, -1 # Set them to some random value
        w = model.linear.weight.item()
        y = model.linear.bias.item()
        # for name, param in model.named_parameters(): # For each named parameter
        #     if param.requires_grad:                  # If it requires gradient (which w and b actually do)
        #         if name == "linear.weight":          # Assign them to respectable variables based on name
        #             w = param.data.item()            # Use .item() to get only the value stored, not the complete Tensor 
        #         if name == "linear.bias":
        #             b = param.data.item()



        print("Epoch: {}, \tW: {}, \tB: {}, \tLoss: {}".format(epoch, w, b, loss.item())) # Print the data
    
    # Calculate the final prediction
    final_prediction = None
    with torch.no_grad(): # Gradients are not needed for testing phase 
        final_prediction = model(X) # Get the final prediction after training 

    # Plot the original data (points) and a line passing through them
    plt.plot(x_data, final_prediction.cpu().data.numpy(), linestyle="-", color="black", label="Predictions", linewidth=2) # Plot the prediction line passing through points
    plt.plot(x_data, y_data, "ro", label="Starting data", alpha=0.5) # Plot the starting data
    plt.xlabel("x") 
    plt.ylabel("y")
    plt.legend() # Show the legend
    plt.grid()   # Show the grid, for ease of comprehension
    plt.show()   # Show the plot!