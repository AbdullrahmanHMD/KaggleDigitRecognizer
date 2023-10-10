# PyTorch imports:
import torch
import torch.nn as nn



class DigitRecognizerModel(nn.Module):

    def __init__(self, input_size=(64, 64), num_classes=10):
        super(DigitRecognizerModel, self).__init__()

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

        kernel_size = (3, 3)
        in_channels, out_channels = 1, 16
        padding, stride = 1, 1
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, padding=padding,
                                stride=stride)

        self.batch_norm_1 = nn.BatchNorm2d(num_features=out_channels)

        kernel_size, stride = 2, 2
        self.max_pool_1 = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

        self.dropout_1 = nn.Dropout(p=0.25)


        kernel_size = (3, 3)
        in_channels, out_channels = 16, 32
        padding, stride = 1, 1
        self.conv_2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, padding=padding,
                                stride=stride)

        self.batch_norm_2 = nn.BatchNorm2d(num_features=out_channels)

        kernel_size, stride = 2, 2
        self.max_pool_2 = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

        self.dropout_2 = nn.Dropout(p=0.35)


        kernel_size = (3, 3)
        in_channels, out_channels = 32, 64
        padding, stride = 1, 1
        self.conv_3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, padding=padding,
                                stride=stride)

        self.batch_norm_3 = nn.BatchNorm2d(num_features=out_channels)

        kernel_size, stride = 2, 2
        self.max_pool_3 = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

        self.dropout_3 = nn.Dropout(p=0.5)

        in_features = 4096
        self.fc = nn.Linear(in_features=in_features, out_features=num_classes)

    def forward(self, x):

        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = self.max_pool_1(x)
        x = self.dropout_1(x)
        x = self.leaky_relu(x)

        x = self.conv_2(x)
        x = self.batch_norm_2(x)
        x = self.max_pool_2(x)
        x = self.dropout_2(x)
        x = self.leaky_relu(x)

        x = self.conv_3(x)
        x = self.batch_norm_3(x)
        x = self.max_pool_3(x)
        x = self.dropout_3(x)
        x = self.leaky_relu(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.leaky_relu(x)

        return x


if __name__ == "__main__":
    model = DigitRecognizerModel()

    print(model)
