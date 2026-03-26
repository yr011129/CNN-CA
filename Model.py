class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(torch.nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


# 双输入CNN-CA
class DualInputCNN(torch.nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # 主
        self.main_branch = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )

        # 次
        self.secondary_branch = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1), # 256*256*16
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),                                                     # 128*128*16
            torch.nn.Conv2d(16, 32, 3, padding=1),                                     # 128*128*32
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),                                                     # 64*64*32
            torch.nn.Conv2d(32, 64, 3, padding=1),                                     # 64*64*64
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),                                                     # 32*32*64
        )

        self.Attention = CoordAtt(inp=128, oup=128)

        self.compress = nn.Conv2d(128, 16, kernel_size=1)


        self.classifier = torch.nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            torch.nn.Flatten(),
            torch.nn.Linear(1024, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(128, 20),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(20, num_classes),
        )

    def forward(self, x1, x2):
        x1_1 = self.main_branch(x1)
        x2_1 = self.secondary_branch(x2)

        x = torch.cat((x1_1, x2_1), dim=1)
        y = self.Attention(x)
        y = self.compress(y)

        return self.classifier(y)


if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 初始化双数据集
    dataset = DualImageDataset(
        main_dir='E:/手势200',
        secondary_dir='E:/手势201',
        transform=transform
    )

    # 获取标签用于分层抽样
    labels = [label for (_, label) in dataset.main_dataset.samples]

    # 分层划分数据集
    train_idx, temp_idx = train_test_split(
        np.arange(len(labels)),
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=[labels[i] for i in temp_idx],
        random_state=42
    )


    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)


    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualInputCNN().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch. optim.Adam(model.parameters(), lr=0.001)