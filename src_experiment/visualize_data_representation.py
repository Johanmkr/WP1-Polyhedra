"""
This is a script for visualizing how a model represents data in its latent layers.
This implementation assumes three-dimensional hidden layers.

Steps:
1. Load a trained model architecture (weights loaded per epoch).
2. Load a fixed 2D inference dataset.
3. Register forward hooks on each hidden layer to capture latent representations.
4. For a given epoch:
   a. Load the corresponding state dict.
   b. Run a forward pass on the dataset.
   c. Store the latent representations for all hidden layers.
5. Visualize the 3D latent representations layer-by-layer.
6. Repeat across epochs and optionally animate the evolution.
"""

# Imports
import torch
import matplotlib.pyplot as plt

# Local imports
if __name__=="__main__":
    from models import get_model
    from dataset import get_new_moons_data
    from paths import get_test_moon_path
else:
    from .models import get_model
    from .dataset import get_new_moons_data
    from .paths import get_test_moon_path
    

class VisualisationOfLatenRepresentation:
    def __init__(self, model_name, dataset_name, noise_level, run_number):
        # Initialise the class by:
        # 1. Loading the model. 
        # 2. Loading the dataset

        # Load model
        self.model = get_model(model_name, run_number)
        
        # Load the data
        self.data = get_new_moons_data("inference", noise_level)
        
        # Basepath
        self.basepath = get_test_moon_path(model_name, dataset_name, noise_level, run_number)
        
        self.epochs = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,124]
        
        
        self.create_hooks()
            
    def create_hooks(self):
        self.latents = {}
        self.labels = {}
        self._current_epoch = None
        self._hook_handles = []

        def make_hook(layer_name):
            def hook(module, input, output):
                self.latents[self._current_epoch][layer_name].append(
                    output.detach().cpu()
                )
            return hook

        for i in range(len(self.model.hidden_sizes)):
            relu = getattr(self.model, f"relu{i + 1}")
            handle = relu.register_forward_hook(make_hook(f"h{i + 1}"))
            self._hook_handles.append(handle)



    def find_latent_rep_for_epoch(self, epoch):
        self.model.load_state_dict(
            torch.load(self.basepath / "state_dicts" / f"epoch{epoch}.pth")
        )

        self.model.eval()
        self._current_epoch = epoch

        self.latents[epoch] = {
            f"h{i + 1}": [] for i in range(len(self.model.hidden_sizes))
        }
        self.labels[epoch] = []

        with torch.no_grad():
            for batch in self.data:
                if isinstance(batch, (list, tuple)):
                    x, y = batch
                else:
                    raise ValueError("DataLoader must return (x, y) for coloring")

                x = x.float()
                _ = self.model(x)

                self.labels[epoch].append(y.cpu())

        # Concatenate batches
        for key in self.latents[epoch]:
            self.latents[epoch][key] = torch.cat(
                self.latents[epoch][key], dim=0
            )

        self.labels[epoch] = torch.cat(self.labels[epoch], dim=0)

            
        
    def visualise_latent_rep_for_epoch(self, epoch):

        latents = self.latents[epoch]
        labels = self.labels[epoch].numpy()

        titles = ["Layer 1", "Layer 2", "Layer 3"]

        fig = plt.figure(figsize=(15, 4))

        for i, key in enumerate(["h1", "h2", "h3"]):
            z = latents[key].numpy()

            ax = fig.add_subplot(1, 3, i + 1, projection="3d")
            scatter = ax.scatter(
                z[:, 0],
                z[:, 1],
                z[:, 2],
                c=labels,
                cmap="viridis",
                s=10,
                alpha=0.8,
            )

            ax.set_title(f"{titles[i]} – Epoch {epoch}")
            ax.set_xlabel("z1")
            ax.set_ylabel("z2")
            ax.set_zlabel("z3")
        plt.tight_layout()
        plt.show()

    def show_epoch(self, epoch):
        self.find_latent_rep_for_epoch(epoch) 
        self.visualise_latent_rep_for_epoch(epoch) 
        
    def animate_for_all_epochs(self):
        # Make an animation for all epochs. 
        pass
        
    def remove_hooks(self):
        for handle in self._hook_handles:
            handle.remove()


if __name__=="__main__":
    obj = VisualisationOfLatenRepresentation("small", "new", 0.05, 0)