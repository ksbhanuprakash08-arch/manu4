import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(5):
            in_c = nf + i * gc
            self.layers.append(nn.Conv2d(in_c, gc, 3, 1, 1))
        self.lff = nn.Conv2d(nf + 5 * gc, nf, 1, 1, 0)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        features = [x]
        for l in self.layers:
            inp = torch.cat(features, 1)
            out = self.relu(l(inp))
            features.append(out)
        out = torch.cat(features, 1)
        out = self.lff(out)
        return out * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.RDB1 = ResidualDenseBlock(nf, gc)
        self.RDB2 = ResidualDenseBlock(nf, gc)
        self.RDB3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class GeneratorRRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4):
        super().__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.RRDB_trunk = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1)

        # upsample blocks (x2 twice for x4)
        upsampler = []
        num_ups = int(scale // 2 if scale % 2 == 0 else 0)
        # For typical x4, we do two x2 upscales
        for _ in range(2):
            upsampler += [
                nn.Conv2d(nf, nf * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        self.upsampler = nn.Sequential(*upsampler)
        self.HR_conv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.RRDB_trunk(fea)
        trunk = self.trunk_conv(trunk)
        fea = fea + trunk
        out = self.upsampler(fea)
        out = self.HR_conv(out)
        out = self.conv_last(out)
        out = torch.sigmoid(out)
        return out


class DiscriminatorVGG(nn.Module):
    def __init__(self, in_nc=3, nf=64):
        super().__init__()
        layers = []
        def conv(in_c, out_c, bn=True):
            if bn:
                return [nn.Conv2d(in_c, out_c, 3, 1, 1), nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2, inplace=True)]
            else:
                return [nn.Conv2d(in_c, out_c, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True)]

        channels = [in_nc, nf, nf, nf*2, nf*2, nf*4, nf*4, nf*8, nf*8]
        for i in range(len(channels)-1):
            layers += conv(channels[i], channels[i+1], bn=(i>0))
            if i % 2 == 1:
                layers += [nn.AvgPool2d(2)]

        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels[-1], 100), nn.LeakyReLU(0.2, inplace=True), nn.Linear(100, 1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class VGGFeatureExtractor(nn.Module):
    def __init__(self, resize=True):
        super().__init__()
        vgg19 = models.vgg19(pretrained=True).features
        self.slice = nn.Sequential(*[vgg19[i] for i in range(35)])
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        # expects input normalized to VGG stats if necessary
        return self.slice(x)


class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = VGGFeatureExtractor()
        self.criterion = nn.MSELoss()

    def forward(self, sr, hr):
        f_sr = self.vgg(sr)
        f_hr = self.vgg(hr)
        return self.criterion(f_sr, f_hr)


class AdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.adv_criterion = nn.BCEWithLogitsLoss()

    def forward(self, pred, target_is_real=True):
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return self.adv_criterion(pred, target)


def train(
    generator,
    discriminator,
    dataloader,
    val_dataloader=None,
    epochs=100,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    save_path="sr_model.pth",
):
    # This is a training loop skeleton demonstrating hybrid loss usage.
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

    content_criterion = nn.MSELoss()
    perceptual_criterion = VGGLoss().to(device)
    adv_criterion = AdversarialLoss().to(device)

    for epoch in range(epochs):
        generator.train(); discriminator.train()
        for i, batch in enumerate(dataloader):
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)

            # ------------------ Train Discriminator ------------------
            with torch.no_grad():
                fake = generator(lr)
            d_optimizer.zero_grad()
            real_out = discriminator(hr)
            fake_out = discriminator(fake.detach())
            d_loss = adv_criterion(real_out, True) + adv_criterion(fake_out, False)
            d_loss.backward()
            d_optimizer.step()

            # ------------------ Train Generator ------------------
            g_optimizer.zero_grad()
            fake = generator(lr)
            # content (pixel-wise)
            content_loss = content_criterion(fake, hr)
            # perceptual (VGG)
            perceptual_loss = perceptual_criterion(fake, hr)
            # adversarial
            pred_fake = discriminator(fake)
            adversarial_loss = adv_criterion(pred_fake, True)

            # hybrid loss
            g_loss = content_loss + 1e-3 * perceptual_loss + 5e-3 * adversarial_loss
            g_loss.backward()
            g_optimizer.step()

        # Save checkpoint each epoch (or intelligently)
        torch.save(generator.state_dict(), save_path)
