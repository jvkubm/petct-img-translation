
import torch
from gan_model import Generator, Discriminator

# Define the device to use (GPU if available, otherwise CPU)
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define input dimensions
batch_size = 1
in_channels = 1
depth = 64
height = 64
width = 64

# Initialize models
generator = Generator(in_channels, in_channels).to(device)
discriminator = Discriminator(in_channels * 2).to(device)

# Create a random tensor to simulate a PET image
pet_image = torch.randn(batch_size, in_channels, depth, height, width).to(device)

# Pass the tensor through the generator to create a CT image
gen_ct_image = generator(pet_image)
print(f"Generated CT image shape: {gen_ct_image.shape}")

# Concatenate PET and generated CT images for the discriminator
input_discriminator = torch.cat((pet_image, gen_ct_image), dim=1)

# Pass the concatenated tensor through the discriminator
discriminator_output = discriminator(input_discriminator)
print(f"Discriminator output shape: {discriminator_output.shape}")