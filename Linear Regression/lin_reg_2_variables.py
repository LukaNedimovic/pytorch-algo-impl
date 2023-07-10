import torch                    # Model creation
import numpy as np              # Mathematics 
import matplotlib.pyplot as plt # Plotting


def forward(X): 
    """Forward pass 
    Based on current value of "w", calculate the prediction for each value of "X"
    This is basically the point of the model - predicting the right "w"

    Args:
        X (torch.Tensor): The values used for making prediction based on current state of "w" and "b"
    """
    return w * X + b # Function is a simple linear equation: f(x) = w * x + b

def calculate_loss(Y_predictions, Y):    
    """ Calculate loss
    
    Based on the made predictions, calculate the error the current "w" produces.
    The lesser the error, the better for the model (therefore the more fitting value of "w").
    Error implemented is the standard Mean Squared Error (compare the error to all values of "X", then take the mean).

    
    Mean Squared Error: 
        error[i] = (Y_predictions[i] - Y)**2
        MSE      = mean(error)

    Args:
        Y_predictions (torch.Tensor): predicted data
        Y             (torch.Tensor): original data
    """
    return torch.mean((Y_predictions - Y) ** 2) 

if __name__ == "__main__":
    # Use GPU, if possible; if not, use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    w = torch.tensor(-0.5, requires_grad=True) # The weight model is trying to predict
    b = torch.tensor(2.0, requires_grad=True)  # The bias model is trying to predict

    # Create data
    x_data = np.arange(-5, 5, 0.4, dtype=np.float32).reshape(-1, 1) # Create a column vector of values
    y_data = 3 * x_data + 2 # The function the model is using to predict w and b
    # y_data = y_data + 0.2 * np.random.rand(x_data.shape[0], x_data.shape[1]) # Optional: Add Gaussian noise
    X = torch.tensor(x_data, dtype=torch.float32, requires_grad=True).to(device) # Create input tensor out of numpy column vector 
    Y = torch.tensor(y_data, dtype=torch.float32, requires_grad=True).to(device) # Create ideal output tensor based on function f(x)
    
    # Plot data
    plt.plot(x_data, y_data, "r+") # Plot point at (X[i], f(X[i]))
    plt.xlabel("x")                # Set the label for x-axis
    plt.ylabel("y")                # Set the label for y-axis
    plt.grid()                     # Show the grid, for ease of comprehension
    plt.show()                     # Show the plot! 

    # Start the training process
    learning_rate = 0.1 # Also called "step", represents how quick we are descending
    epochs        = 30  # Train model for "30" epochs (also called "iterations")
    
    for epoch in range(epochs):
        # Forward pass
        Y_predictions = forward(X)              # Predict data based on current value of "w"
        
        # Calculate the loss
        loss = calculate_loss(Y_predictions, Y) # Calculate the loss based on predictions 
        
        # Backward pass
        loss.backward()                               # Calculate the gradient w.r.t "w" and "b"      
        w.data = w.data - learning_rate * w.grad.data # Make a step opposite of gradient to make error smaller
        b.data = b.data - learning_rate * b.grad.data 
        w.grad.data.zero_()                           # Zero out the gradients
        b.grad.data.zero_()                           

        print("Epoch: {}, \tW: {}, \tB: {}, \tLoss {}:".format(epoch, w.item(), b.item(), loss.item())) # Print the data


    # Calculate the final prediction
    final_prediction = w.item() * x_data + b.item() # "w" and "b" now have the final predicted values

    # Plot the original data (points) and a line passing through them
    plt.plot(x_data, final_prediction, linestyle="-", color="black", label="Predictions", linewidth=2) # Plot the prediction line passing through points
    plt.plot(x_data, y_data, "ro", label="Starting data", alpha=0.5) # Plot the starting data
    plt.xlabel("x") 
    plt.ylabel("y")
    plt.legend() # Show the legend
    plt.grid()   # Show the grid, for ease of comprehension
    plt.show()   # Show the plot!
