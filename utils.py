import matplotlib.pyplot as plt
from inspect import isfunction


def plot_imgs(inputs,name:str, dir:str='imgs'):
    fig, axes = plt.subplots(nrows=1, ncols=inputs.size(0), figsize=(16, 16))
    for idx in range(inputs.size(0)):
        axes[idx].imshow(inputs[idx].squeeze().numpy(), cmap='gray')
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
    plt.tight_layout()
    plt.savefig(f'{dir}/{name}.png')
    
def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d