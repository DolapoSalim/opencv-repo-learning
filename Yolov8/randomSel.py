import os
import random

def pick_random_image(directory):
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')
    images = [f for f in os.listdir(directory) if f.lower().endswith(image_extensions)]
    
    if not images:
        print("No images found in the directory.")
        return None
    
    return random.choice(images)

# Example usage
directory = "PATH_TO_DIRECTORY"
random_image = pick_random_image(directory)

if random_image:
    print("Randomly selected image:", random_image)
