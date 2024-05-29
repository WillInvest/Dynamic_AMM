from PIL import Image
import glob
import os
name = 'AMM_2024-05-28_20-30-13_DDPG_plot'
# File paths for the images (assuming they are named sequentially and saved in the specified directory)
file_paths = sorted(
    glob.glob(f'/Users/haofu/AMM/AMM-Python/src/env/saved_plot/{name}/plot_model_*.png'),
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1])
)

# Filter file paths to only include those in the specified range
file_paths = [fp for fp in file_paths if 1024 <= int(os.path.splitext(os.path.basename(fp))[0].split('_')[-1]) <= 299008]

# Verify that file paths are correctly identified
if not file_paths:
    print("No files found. Please check the directory and file naming pattern.")
else:
    print(f"Found {len(file_paths)} files.")

# Verify that images are being loaded correctly
images = []
for file_path in file_paths:
    try:
        img = Image.open(file_path)
        images.append(img)
        print(f"Successfully loaded {file_path}")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

# Create and save the GIF if images were loaded
if images:
    gif_path = '/Users/haofu/AMM/AMM-Python/src/env/convergence.gif'
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=10, loop=0)
    print(f"GIF saved to {gif_path}")
else:
    print("No images loaded. GIF creation skipped.")
