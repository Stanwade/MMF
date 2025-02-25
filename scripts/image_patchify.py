import os
from PIL import Image
import numpy as np
import math

def image_patchify(folder_path: str, patch_size: int, save_patches: bool = False):
    """
    Given a folder path and a patch size, this function reads all the images in the folder, resizes them to be a multiple of the patch size, and divides them into smaller patches of the specified size. The function returns a list of images, where each image is a list of patches.

    Parameters:
        folder_path (str): The path to the folder containing the images.
        patch_size (int): The size of the patches to be extracted from the images.

    Returns:
        list: A list of images, where each image is a list of patches.
    """
    images = []
    for image_path in os.listdir(folder_path):
        
        if not (image_path.endswith('.png') or image_path.endswith('.jpg')):
            continue
        
        image = Image.open(os.path.join(folder_path, image_path))
        image = image.resize((math.ceil(image.width / patch_size) * patch_size, math.ceil(image.height / patch_size) * patch_size), Image.NEAREST)
        image_array = np.asarray(image)
        num_of_patch_rows = image_array.shape[0] // patch_size
        num_of_patch_cols = image_array.shape[1] // patch_size
        patches = []
        for row in range(num_of_patch_rows):
            for col in range(num_of_patch_cols):
                patch = image_array[row * patch_size:(row + 1) * patch_size, col * patch_size:(col + 1) * patch_size]
                patches.append({'patch': patch, 'coord':(row,col)})
                
                if save_patches:
                    if not os.path.exists(f'{folder_path}/patches'):
                        os.mkdir(f'{folder_path}/patches')
                    Image.fromarray(patch).save(f'{folder_path}/patches/{image_path.split(".")[0]}_{row}_{col}.png')
        
        images.append(patches)
    return images

if __name__ == '__main__':
    patch_size = 32
    test_x = np.random.randint(0,256,(1,32*3,32*2))
    
    image = Image.fromarray(test_x[0])
    image = image.resize((math.ceil(image.width / patch_size) * patch_size, math.ceil(image.height / patch_size) * patch_size), Image.NEAREST)
    image_array = np.asarray(image)
    
    num_of_patch_rows = image_array.shape[0] // patch_size
    print(f'num_of_patch_rows = {num_of_patch_rows}')
    num_of_patch_cols = image_array.shape[1] // patch_size
    print(f'num_of_patch_cols = {num_of_patch_cols}')
    images = []
    patches = []
    for row in range(num_of_patch_rows):
        for col in range(num_of_patch_cols):
            patch = image_array[row * patch_size:(row + 1) * patch_size, col * patch_size:(col + 1) * patch_size]
            patches.append({'patch': patch, 'coord':(row,col)})
    images.append(patches)
    
    patches_coords = [parch['coord'] for parch in patches]
    print(patches_coords)
    
    print('testing if patches is right')
    print(test_x[0,:patch_size,:patch_size])
    print('patches 0: ')
    print(patches[0]['patch'])
    print('bool check')
    print(test_x[0,:patch_size, :patch_size] == patches[0]['patch'])


