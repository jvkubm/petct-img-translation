import os 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from gan_model import Generator, Discriminator
from my_dataset import TranslDataset2D
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

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
checkpoint_dir = '/mnt/data/mij17663/nac2ac_sep/model2D/checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# Directory to save the generated images
output_image_dir = '/mnt/data/mij17663/nac2ac_sep/model2D/generated_images'
os.makedirs(output_image_dir, exist_ok=True)

# Define hyperparameters
num_epochs = 100
batch_size = 1
learning_rate = 0.0002
in_channels = 1
out_channels = 1
is_3d = False

# Initialize models
generator = Generator(in_channels, out_channels, is_3d=is_3d).to(device)
discriminator = Discriminator(in_channels + out_channels, is_3d=is_3d).to(device)

# Loss functions
criterion_GAN = nn.BCELoss()  # Binary cross entropy loss for GAN, there is also nn.BCEWithLogitsLoss with sigmoid included
criterion_pixelwise = nn.L1Loss()  # L1 loss for pixel-wise comparison, L1 is Mean Absolute Error

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Dataset and DataLoader creation
generator1 = torch.Generator().manual_seed(42)
image_dir = '/mnt/data/mij17663/nac2ac_sep/2d_slices/NAC_PET_Tr'
labels_dir = '/mnt/data/mij17663/nac2ac_sep/2d_slices/AC_PET_Tr'
dataset = TranslDataset2D(image_dir, labels_dir)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.85, 0.15], generator1)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Print model structure
print(f"Model structure: {generator}\n\n")

for name, param in generator.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

# Training loop
# ToDo: do I need model.train() here?
# whats the diff between model.train() and model.zero_grad() here?
for epoch in range(num_epochs):
    generator.train()
    discriminator.train()
    for i, (pet_images, ct_images) in enumerate(train_loader):
        pet_images = pet_images.float().to(device)
        ct_images = ct_images.float().to(device)

        # Adversarial ground truths
        if is_3d:
            real = torch.ones((pet_images.size(0), 1, 1, 1, 1), requires_grad=False).to(device)
            fake = torch.zeros((pet_images.size(0), 1, 1, 1, 1), requires_grad=False).to(device)
        else:
            real = torch.ones((pet_images.size(0), 1, 1, 1), requires_grad=False).to(device)
            fake = torch.zeros((pet_images.size(0), 1, 1, 1), requires_grad=False).to(device)
        # ---------------------
        #  Train Generator
        # ---------------------
        optimizer_G.zero_grad()

        # Generate a batch of images
        gen_ct_images = generator(pet_images)
        print(f"Generated CT image shape: {gen_ct_images.shape}")

        # Loss measures generator's ability to fool the discriminator
        # generator's goal is to fool the discriminator into thinking that generated image is real
        # so we compare only to the label for real images (if discriminator output is close to 1 then generator is doing well)
        input_discriminator = torch.cat((pet_images, gen_ct_images), 1)
        loss_GAN = criterion_GAN(discriminator(input_discriminator), real) # Compare the discriminator output with real labels
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
        if i % 100 == 0:
            print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(train_loader)}] [D loss: {loss_D.item()}] [G loss: {loss_G.item()}]")

    # Save the model checkpoints after every 10 epochs
    if (epoch+1) % 10 == 0:
        torch.save(generator.state_dict(), os.path.join(checkpoint_dir, f'generator_epoch_{epoch}.pth'))
        torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, f'discriminator_epoch_{epoch}.pth'))
        print(f"Saved model checkpoints for epoch {epoch}")
    
    # Validation loop
    generator.eval()
    ssim_total = 0
    rmse_total = 0
    psnr_total = 0
    num_val_samples = 0
    with torch.no_grad():
        for j, (val_pet_images, val_ct_images) in enumerate(val_loader):
            val_pet_images = val_pet_images.float().to(device)
            val_ct_images = val_ct_images.float().to(device)

            # Generate CT images from PET images
            val_gen_ct_images = generator(val_pet_images)

            # Save some generated images alongside real images for validation
            if j < 5:  # Save the first 5 images
                save_image(val_pet_images, os.path.join(output_image_dir, f'epoch_{epoch}_val_{j}_pet.png'))
                save_image(val_ct_images, os.path.join(output_image_dir, f'epoch_{epoch}_val_{j}_real_ct.png'))
                save_image(val_gen_ct_images, os.path.join(output_image_dir, f'epoch_{epoch}_val_{j}_gen_ct.png'))

            # Compute evaluation metrics
            val_pet_images_np = val_pet_images.cpu().numpy().squeeze()
            val_ct_images_np = val_ct_images.cpu().numpy().squeeze()
            val_gen_ct_images_np = val_gen_ct_images.cpu().numpy().squeeze()

            ssim_value = ssim(val_ct_images_np, val_gen_ct_images_np, data_range=val_gen_ct_images_np.max() - val_gen_ct_images_np.min())
            rmse_value = np.sqrt(np.mean((val_ct_images_np - val_gen_ct_images_np) ** 2))
            psnr_value = psnr(val_ct_images_np, val_gen_ct_images_np, data_range=val_gen_ct_images_np.max() - val_gen_ct_images_np.min())

            ssim_total += ssim_value
            rmse_total += rmse_value
            psnr_total += psnr_value
            num_val_samples += 1

    # Compute average metrics
    ssim_avg = ssim_total / num_val_samples
    rmse_avg = rmse_total / num_val_samples
    psnr_avg = psnr_total / num_val_samples

    print(f"Validation complete for epoch {epoch}")
    print(f"SSIM: {ssim_avg:.4f}, RMSE: {rmse_avg:.4f}, PSNR: {psnr_avg:.4f}")