import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Generator
# -------------------------------
class Generator(nn.Module):
    def __init__(self, noise_dim, seq_len, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_len)
        )
    def forward(self, z):
        return self.net(z)

# -------------------------------
# Discriminator
# -------------------------------
class Discriminator(nn.Module):
    def __init__(self, seq_len, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.LeakyReLU(0.2),   # ✅ LeakyReLU instead of ReLU
            nn.Linear(hidden_dim, 1)
            # ❌ No Sigmoid (we’ll use BCEWithLogitsLoss)
        )
    def forward(self, x):
        return self.net(x)

# -------------------------------
# Training Function
# -------------------------------
def run_gan(input_csv, output_path, seq_len=30, noise_dim=10, hidden_dim=64, batch_size=32, epochs=500, lr=0.0002):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load data
    df = pd.read_csv(input_csv)
    prices = df['S&P500'].values
    sequences = []
    for i in range(len(prices)-seq_len):
        sequences.append(prices[i:i+seq_len])
    sequences = torch.FloatTensor(sequences)

    # Initialize models
    generator = Generator(noise_dim, seq_len, hidden_dim)
    discriminator = Discriminator(seq_len, hidden_dim)
    criterion = nn.BCEWithLogitsLoss()   # ✅ replaces Sigmoid+BCELoss
    opt_gen = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))  # ✅ stable GAN trick
    opt_disc = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Training loop
    for epoch in range(epochs):
        for i in range(0, len(sequences), batch_size):
            real = sequences[i:i+batch_size]
            batch_size_actual = real.size(0)

            # Label smoothing + noise ✅
            real_labels = torch.ones(batch_size_actual, 1) * 0.9  
            fake_labels = torch.zeros(batch_size_actual, 1) + 0.1  

            # Train Discriminator
            z = torch.randn(batch_size_actual, noise_dim)
            fake = generator(z)
            disc_real = discriminator(real)
            disc_fake = discriminator(fake.detach())
            loss_disc = criterion(disc_real, real_labels) + criterion(disc_fake, fake_labels)
            opt_disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            # Train Generator
            disc_fake = discriminator(fake)
            loss_gen = criterion(disc_fake, torch.ones(batch_size_actual, 1))  # wants fakes = real
            opt_gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss D: {loss_disc.item():.4f}, Loss G: {loss_gen.item():.4f}")

    # Generate sequences
    z = torch.randn(5, noise_dim)
    synthetic = generator(z).detach().numpy()

    # Plot
    plt.figure(figsize=(10,6))
    for seq in synthetic:
        plt.plot(seq)
    plt.title("Generated Stock Sequences (GAN)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"GAN synthetic sequences plot saved at: {output_path}")

    # Save sequences to CSV
    csv_path = os.path.join(os.path.dirname(output_path), "gan_synthetic.csv")
    pd.DataFrame(synthetic).to_csv(csv_path, index=False)
    print(f"GAN synthetic sequences saved at: {csv_path}")
