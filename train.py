import os 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from gan_model import Generator, Discriminator

# Determine the device to use (GPU if available, otherwise CPU)
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Directory to save the checkpoints
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# Define hyperparameters
num_epochs = 100
batch_size = 4
learning_rate = 0.0002
in_channels = 1
out_channels = 1

# Initialize models
generator = Generator(in_channels, out_channels).to(device)
discriminator = Discriminator(in_channels + out_channels).to(device)

# Loss functions
criterion_GAN = nn.BCELoss()  # Binary cross entropy loss for GAN, there is also nn.BCEWithLogitsLoss with sigmoid included
criterion_pixelwise = nn.L1Loss()  # L1 loss for pixel-wise comparison, L1 is Mean Absolute Error

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# DataLoader (replace with your dataset)
train_loader = DataLoader(..., batch_size=batch_size, shuffle=True)

# Print model structure
print(f"Model structure: {generator}\n\n")

for name, param in generator.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

# Training loop
# ToDo: do I need model.train() here?
# whats the diff between model.train() and model.zero_grad() here?
for epoch in range(num_epochs):
    for i, (pet_images, ct_images) in enumerate(train_loader):
        pet_images = pet_images.to(device)
        ct_images = ct_images.to(device)

        # Adversarial ground truths
        real = torch.ones((pet_images.size(0), 1, 1, 1, 1), requires_grad=False).to(device)
        fake = torch.zeros((pet_images.size(0), 1, 1, 1, 1), requires_grad=False).to(device)

        # ---------------------
        #  Train Generator
        # ---------------------
        optimizer_G.zero_grad()

        # Generate a batch of images
        gen_ct_images = generator(pet_images)

        # Loss measures generator's ability to fool the discriminator
        # generator's goal is to fool the discriminator into thinking that generated image is real
        # so we compare only to the label for real images (if discriminator output is close to 1 then generator is doing well) 
        loss_GAN = criterion_GAN(discriminator(torch.cat((pet_images, gen_ct_images), 1)), real) # Compare the discriminator output with real labels
        loss_pixel = criterion_pixelwise(gen_ct_images, ct_images)

        # Total generator loss
        loss_G = loss_GAN + 100 * loss_pixel

        loss_G.backward() # Backpropagate the loss
        optimizer_G.step() # Update the generator weights

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Real loss, we want that discriminator correctly classifies real images as real
        loss_real = criterion_GAN(discriminator(torch.cat((pet_images, ct_images), 1)), real)
        # Fake loss, we want that discriminator correctly classifies fake images as fake
        loss_fake = criterion_GAN(discriminator(torch.cat((pet_images, gen_ct_images.detach()), 1)), fake)

        # Total discriminator loss
        loss_D = (loss_real + loss_fake) / 2 # Average the real and fake loss
        loss_D.backward() # Backpropagate the loss
        optimizer_D.step() # Update the discriminator weights

        # Print the losses
        print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(train_loader)}] [D loss: {loss_D.item()}] [G loss: {loss_G.item()}]")

    # Save the model checkpoints after every 10 epochs
    if (epoch+1) % 10 == 0:
        torch.save(generator.state_dict(), os.path.join(checkpoint_dir, f'generator_epoch_{epoch}.pth'))
        torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, f'discriminator_epoch_{epoch}.pth'))
        print(f"Saved model checkpoints for epoch {epoch}")
    
    # here it would be nice to either to either show the results on the validation set
    # or somehow qualitatively show results, like print images or something 