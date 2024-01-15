import torch
import torch.nn as nn

class MedicalImageEncoder(nn.Module):
    def __init__(self, encoded_space_dim, input_channels=1):
        super().__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(input_channels, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(64 * (input_size // 16) * (input_size // 16), encoded_space_dim)
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        return x

# Example usage:
# Define your image size, for example, 256 x 256 pixels
input_size = 256
# Assuming grayscale medical images, so input_channels=1
encoder = MedicalImageEncoder(encoded_space_dim=128, input_channels=1)
# Create a dummy batch of images with shape (batch_size, channels, height, width)
# For a single image, you can use batch_size=1
dummy_images = torch.randn(4, 1, input_size, input_size)
# Forward pass through the encoder
encoded_images = encoder(dummy_images)
print('Encoded images shape:', encoded_images.shape)
