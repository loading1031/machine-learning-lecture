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

# Discriminator (다운스케일링 역할만 수행)
class Discriminator:
    def __init__(self):
        pass

    def __call__(self, high_res, low_res):
        # 고해상도를 저해상도로 다운스케일링
        scaled_low_res = nn.functional.interpolate(high_res, size=low_res.shape[-2:], mode='bicubic', align_corners=True)
        return scaled_low_res


# 데이터 준비
transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 모델 생성
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 손실 함수 및 옵티마이저
criterion = nn.BCELoss()

# Discriminator 인스턴스 생성 (학습하지 않음)
discriminator = Discriminator()

# 점진적 학습 단계 정의
stages = [(2, 14), (2, 28), (2, 64), (2, 128)]  # (업스케일링 배율, 해상도)
for upscale_factor, resolution in stages:
    print(f"Starting stage: {resolution}x{resolution}")

    generator = Generator(upscale_factor=upscale_factor).to(device)
    gen_optimizer = optim.Adam(generator.parameters(), lr=1e-4)

    for epoch in range(5):  # 각 단계에서 5 에포크 학습
        for batch_idx, (data, _) in enumerate(data_loader):
            # 데이터 준비
            data = data.to(device)  # 원본 데이터 (28x28)
            low_res = nn.functional.interpolate(data, size=(resolution // upscale_factor, resolution // upscale_factor), mode='bicubic')

            # Generator 생성
            fake_high_res = generator(low_res)

            # Discriminator로 다운스케일링 (학습 없음)
            compressed_low_res = discriminator(fake_high_res, low_res)

            # 손실 계산: 압축된 고해상도와 원본 저해상도의 차이
            g_loss = nn.MSELoss()(compressed_low_res, low_res)

            # Generator 학습
            gen_optimizer.zero_grad()
            g_loss.backward()
            gen_optimizer.step()

            # 학습 상태 출력
            if batch_idx % 100 == 0:
                print(f"Stage: {resolution}x{resolution}, Epoch [{epoch+1}/5] Batch [{batch_idx}/{len(data_loader)}] "
                      f"G Loss: {g_loss.item():.4f}")

    # 결과 확인
    data_iter = iter(data_loader)
    data, _ = next(data_iter)
    data = data.to(device)
    low_res = nn.functional.interpolate(data, size=(resolution // upscale_factor, resolution // upscale_factor), mode='bicubic')
    fake_high_res = generator(low_res).detach().cpu()
    compressed_low_res = discriminator(fake_high_res.to(device), low_res).detach().cpu()

    plt.figure(figsize=(15, 5))
    for i in range(5):
        # 저해상도
        plt.subplot(3, 5, i + 1)
        plt.imshow(low_res[i].squeeze().cpu().numpy(), cmap='gray')
        plt.title("Low Res")
        plt.axis('off')

        # 고해상도 (Generator 출력)
        plt.subplot(3, 5, i + 6)
        plt.imshow(fake_high_res[i].squeeze().numpy(), cmap='gray')
        plt.title(f"Super Res ({resolution}x{resolution})")
        plt.axis('off')

        # 압축된 고해상도
        plt.subplot(3, 5, i + 11)
        plt.imshow(compressed_low_res[i].squeeze().numpy(), cmap='gray')
        plt.title("Compressed Low Res")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
