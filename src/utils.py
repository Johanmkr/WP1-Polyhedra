# Utility file
import matplotlib.cm as cm
import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt

# Shapes
# input (1, 1000, 2)
# target (1, 1000)
# layer1 (EPOCHS, 1000, 3)
# layer2 (EPOCHS, 1000, 2)
# output (EPOCHS, 1000)

def visualize_training(losses, input, layer1, layer2, output, target):
    EPOCH = -1
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()
    ax2.remove()
    ax2 = fig.add_subplot(222, projection='3d')
    losses = losses.reshape(losses.shape[0], -1) # (EPOCHS, 1)
    output = output.reshape(output.shape[0], output.shape[1],-1) # (EPOCHS, 1000, 1)
    ax1.scatter(input[0,:,0], input[0,:,1], c=target)
    ax1.set_title('Input')
    
    ax2.scatter(layer1[EPOCH,:,0], layer1[EPOCH,:,1], layer1[EPOCH,:,2], c=target)
    ax2.set_title('Layer 1')
    ax3.scatter(layer2[EPOCH,:,0], layer2[EPOCH,:,1], c=target)
    ax3.set_title('Layer 2')
    ax4.scatter(output[EPOCH], np.zeros_like(output[EPOCH]), c=target)
    ax4.set_title('Output')
    
    
    plt.show()
    
if __name__ == "__main__":
    try:
        losses = np.load('losses.npy')
        input = np.load('input.npy')
        target = np.load('target.npy')
        layer1 = np.load('layer1.npy')
        layer2 = np.load('layer2.npy')
        output = np.load('output.npy')
        visualize_training(losses, input, layer1, layer2, output, target)
    except FileNotFoundError:
        print("Data files not found. Please run the training script first.")
        # Optionally, you can call a function to train the model here
        # train_model() or similar
        # For now, we just exit
        exit(1)