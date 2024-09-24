import warnings

warnings.filterwarnings('ignore')

# Load model directly
from transformers import AutoModel

# model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")

# for param in model.parameters():
#     param.data = param.data.contiguous()

# model.save_pretrained('./openai/clip-vit-base-patch32')

model = AutoModel.from_pretrained("openai/clip-vit-large-patch14")

for param in model.parameters():
    param.data = param.data.contiguous()

model.save_pretrained('./openai/clip-vit-large-patch14')