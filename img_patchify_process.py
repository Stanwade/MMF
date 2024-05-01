from scripts.image_patchify import image_patchify

if __name__ == '__main__':
    image_patchify('imgs', 128, save_patches=True)