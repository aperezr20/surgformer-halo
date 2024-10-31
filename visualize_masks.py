import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from PIL import Image

# Define the mapping of object IDs to names
object_mapping = {
    0: "a_machine",
    1: "a_mask",
    2: "arm_board_l",
    3: "arm_board_r",
    4: "door",
    5: "human",
    6: "iv_bag",
    7: "lift",
    8: "or_table",
    9: "p_ox",
    10: "scanner",
    11: "sliding_board",
    12: "stretcher",
    13: "wheelchair"
}

def visualize_image_with_masks_and_legend(image_dir, mask_dir, output_dir, legend_output_dir):
    # Create output directories if they don't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(legend_output_dir).mkdir(parents=True, exist_ok=True)
    
    # Define colors for classes
    colors = list(mcolors.TABLEAU_COLORS.values())
    
    # Loop through each .npz file in the mask directory
    for npz_file in os.listdir(mask_dir):
        if npz_file.endswith(".npz"):
            mask_path = os.path.join(mask_dir, npz_file)
            image_path = os.path.join(image_dir, npz_file.replace('.npz', '.png'))
            
            # Load the mask and image
            masks = np.load(mask_path)['masks']  # assuming the array is saved under 'arr_0'
            image = np.array(Image.open(image_path).convert('RGB'))
            H, W, N = masks.shape
            
            # Initialize a colored mask with 3 channels for RGB
            colored_mask = np.zeros((H, W, 3), dtype=np.uint8)
            
            # Apply a color to each class
            for class_idx in range(N):
                class_color = np.array(mcolors.to_rgb(colors[class_idx % len(colors)])) * 255
                colored_mask[masks[:, :, class_idx] == 1] = class_color
            
            # Blend the original image and the colored mask
            overlay = np.clip(image * 0.6 + colored_mask * 0.4, 0, 255).astype(np.uint8)
            
            # Save the overlayed image in the output directory
            output_path = os.path.join(output_dir, npz_file.replace('.npz', '.png'))
            overlay_image = Image.fromarray(overlay)
            overlay_image.save(output_path)
            
            # Create a figure for the overlay with a legend
            fig, ax = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 1]})
            ax[0].imshow(overlay)
            ax[0].axis('off')
            ax[0].set_title('Overlayed Image')
            
            # Legend on the right
            for class_idx in range(N):
                if class_idx in object_mapping:
                    class_color = mcolors.to_rgb(colors[class_idx % len(colors)])
                    ax[1].plot([], [], marker='o', markersize=10, color=class_color, label=object_mapping[class_idx])
            
            ax[1].legend(loc='center', fontsize=8, title="Classes")
            ax[1].axis('off')
            ax[1].set_title('Legend')
            
            # Save the figure with the legend
            legend_output_path = os.path.join(legend_output_dir, npz_file.replace('.npz', '_with_legend.png'))
            plt.savefig(legend_output_path, bbox_inches='tight', pad_inches=0, transparent=True)
            plt.close(fig)
            
# Usage
image_directory = '/home/han/Documents/github/halo_project/dataset/dataset_keyframe/case_1'
mask_directory = '/home/aperezr/halo/Surgformer/data/halo/masks/case_1'
output_directory = '/home/aperezr/halo/Surgformer/data/halo/mask_visuals/case_1'
legend_output_directory = 'path/to/output/overlayed_images_with_legend'
visualize_image_with_masks_and_legend(image_directory, mask_directory, output_directory, legend_output_directory)
  