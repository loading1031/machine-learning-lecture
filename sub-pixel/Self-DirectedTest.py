import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Sub-pixel Convolution Layer 정의
class SubPixelConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor):
        super(SubPixelConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        return self.pixel_shuffle(self.conv(x))

# Generator (ESPCN)
class Generator(nn.Module):
    def __init__(self, upscale_factor):
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            SubPixelConv2d(32, 1, upscale_factor)
        )

    def forward(self, x):
        return self.layers(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.downscale = nn.Upsample(scale_factor=0.125, mode='bicubic', align_corners=True)  # 128×128 → 14×14
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(14 * 14, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, high_res, low_res):
        # 고해상도 이미지를 저해상도로 변환
        scaled_low_res = self.downscale(high_res)
        diff = torch.abs(scaled_low_res - low_res)  # 원본 저해상도와 차이 계산
        return self.fc(self.flatten(diff))

# 데이터 준비
transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 모델 생성
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator(upscale_factor=8).to(device)  # 14×14 → 128×128
discriminator = Discriminator().to(device)

# 손실 함수 및 옵티마이저
criterion = nn.BCELoss()
gen_optimizer = optim.Adam(generator.parameters(), lr=1e-4)
disc_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4)

# 학습
epochs = 10
for epoch in range(epochs):
    for batch_idx, (data, _) in enumerate(data_loader):
        # 데이터 준비
        data = data.to(device)  # 원본 데이터 (28x28)
        low_res = nn.functional.interpolate(data, scale_factor=0.5, mode='bicubic')  # 14x14 다운스케일링

        # Generator 생성
        fake_high_res = generator(low_res)

        # Discriminator 학습
        real_labels = torch.ones(low_res.size(0), 1).to(device)
        fake_labels = torch.zeros(low_res.size(0), 1).to(device)

        # 진짜 데이터 학습 (저해상도와 Ground Truth)
        real_output = discriminator(data, low_res)
        d_loss_real = criterion(real_output, real_labels)

        # 가짜 데이터 학습 (저해상도와 Generator 출력)
        fake_output = discriminator(fake_high_res.detach(), low_res)
        d_loss_fake = criterion(fake_output, fake_labels)

        # 총 Discriminator 손실
        d_loss = d_loss_real + d_loss_fake
        disc_optimizer.zero_grad()
        d_loss.backward()
        disc_optimizer.step()

        # Generator 학습
        fake_output = discriminator(fake_high_res, low_res)
        g_loss = criterion(fake_output, real_labels)  # Generator는 "진짜"로 판별되도록 학습

        gen_optimizer.zero_grad()
        g_loss.backward()
        gen_optimizer.step()

        # 학습 상태 출력
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(data_loader)}] "
                  f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

# 결과 시각화
data_iter = iter(data_loader)
data, _ = next(data_iter)
data = data.to(device)
low_res = nn.functional.interpolate(data, scale_factor=0.5, mode='bicubic')
fake_high_res = generator(low_res).detach().cpu()

plt.figure(figsize=(15, 5))
for i in range(5):
    # 저해상도
    plt.subplot(3, 5, i + 1)
    plt.imshow(low_res[i].squeeze().cpu().numpy(), cmap='gray')
    plt.title("Low Res")
    plt.axis('off')

    # 고해상도 (Ground Truth)
    plt.subplot(3, 5, i + 6)
    plt.imshow(data[i].squeeze().cpu().numpy(), cmap='gray')
    plt.title("Ground Truth")
    plt.axis('off')

    # Generator 출력
    plt.subplot(3, 5, i + 11)
    plt.imshow(fake_high_res[i].squeeze().numpy(), cmap='gray')
    plt.title("Super Res")
    plt.axis('off')

plt.tight_layout()
plt.show()
