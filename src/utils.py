import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np

def animate_training(trainer, interval=50, everyth_epoch=50):
    """
    Creates an animation of the training process for a MoonTrainer object.
    
    Parameters:
        trainer (MoonTrainer): The MoonTrainer object containing training data.
        interval (int): Interval between frames in milliseconds.
    """
    # Extract data from the trainer
    x = trainer.x.cpu().numpy()
    y = trainer.y.cpu().numpy()
    hidden_layers = trainer.hidden_layers
    outputs = trainer.output
    losses = trainer.losses
    epochs = trainer.trained_epochs
    # Determine if the model has one or two hidden layers
    has_two_hidden_layers = len(hidden_layers[0]) == 2
    # Create figure and subplots
    fig = plt.figure(figsize=(12, 10))
    ax_loss = fig.add_axes([0.1, 0.85, 0.8, 0.1])  # Loss plot spanning the top
    # Subplots for layers
    if has_two_hidden_layers:
        ax_input = fig.add_subplot(221)
        ax_hidden1 = fig.add_subplot(222, projection='3d')
        ax_hidden2 = fig.add_subplot(223, projection='3d')
        ax_output = fig.add_subplot(224)
    else:
        ax_input = fig.add_subplot(131)
        ax_hidden1 = fig.add_subplot(132, projection='3d')
        ax_output = fig.add_subplot(133)
        ax_hidden2 = None  # No second hidden layer
    # Initialize plots
    def init():
        ax_loss.clear()
        ax_input.clear()
        ax_hidden1.clear()
        if has_two_hidden_layers:
            ax_hidden2.clear()
        ax_output.clear()
        # Loss plot
        ax_loss.set_title('Loss vs Epochs')
        ax_loss.set_xlabel('Epochs')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_xlim(0, epochs)
        ax_loss.set_ylim(0, max(losses) * 1.1)
        # Input layer
        ax_input.scatter(x[:, 0], x[:, 1], c=y, cmap='viridis')
        ax_input.set_title('Input Layer')
        ax_input.set_xticks([])
        ax_input.set_yticks([])
        # Hidden layers
        ax_hidden1.set_title('Hidden Layer 1')
        ax_hidden1.set_xticks([])
        ax_hidden1.set_yticks([])
        ax_hidden1.set_zticks([])
        if has_two_hidden_layers:
            ax_hidden2.set_title('Hidden Layer 2')
            ax_hidden2.set_xticks([])
            ax_hidden2.set_yticks([])
            ax_hidden2.set_zticks([])
        # Output layer
        ax_output.set_title('Output Layer')
        ax_output.set_xticks([])
        ax_output.set_yticks([])
        return fig,
    # Update function for animation
    def update(epoch):
        ax_loss.clear()
        ax_input.clear()
        ax_hidden1.clear()
        if has_two_hidden_layers:
            ax_hidden2.clear()
        ax_output.clear()
        # Loss plot
        ax_loss.plot(range(epoch + 1), losses[:epoch + 1], color='blue')
        ax_loss.set_title('Loss vs Epochs')
        ax_loss.set_xlabel('Epochs')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_xlim(0, epochs)
        ax_loss.set_ylim(0, max(losses) * 1.1)
        # Input layer
        ax_input.scatter(x[:, 0], x[:, 1], c=y, cmap='viridis')
        ax_input.set_title('Input Layer')
        ax_input.set_xticks([])
        ax_input.set_yticks([])
        # Hidden layer 1
        hidden_data = hidden_layers[epoch]
        h1 = hidden_data[0]
        if h1.shape[1] == 2:  # 2D hidden layer
            ax_hidden1.scatter(h1[:, 0], h1[:, 1], c=y, cmap='viridis')
        elif h1.shape[1] == 3:  # 3D hidden layer
            ax_hidden1.scatter(h1[:, 0], h1[:, 1], h1[:, 2], c=y, cmap='viridis')
        ax_hidden1.set_title('Hidden Layer 1')
        ax_hidden1.set_xticks([])
        ax_hidden1.set_yticks([])
        ax_hidden1.set_zticks([])
        # Hidden layer 2 (if applicable)
        if has_two_hidden_layers:
            h2 = hidden_data[1]
            if h2.shape[1] == 2:  # 2D hidden layer
                ax_hidden2.scatter(h2[:, 0], h2[:, 1], c=y, cmap='viridis')
            elif h2.shape[1] == 3:  # 3D hidden layer
                ax_hidden2.scatter(h2[:, 0], h2[:, 1], h2[:, 2], c=y, cmap='viridis')
            ax_hidden2.set_title('Hidden Layer 2')
            ax_hidden2.set_xticks([])
            ax_hidden2.set_yticks([])
            ax_hidden2.set_zticks([])
        # Output layer
        out = outputs[epoch]
        ax_output.scatter(out, np.zeros_like(out), c=y, cmap='viridis')
        ax_output.set_title('Output Layer')
        ax_output.set_xticks([])
        ax_output.set_yticks([])
        return fig,
    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=range(0, epochs, everyth_epoch), init_func=init, interval=interval, blit=True)
    # Save animation
    save_path = f"animations/{trainer.model.name}_epochs_{epochs}.mp4"
    ani.save(save_path, writer='ffmpeg')
    plt.show()
    
if __name__ == "__main__":
    pass