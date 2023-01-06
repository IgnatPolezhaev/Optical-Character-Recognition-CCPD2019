import torch
import torchvision
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.block_0 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1), # b_sx1x128x256 -> b_sx64x128x256
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2)), # b_sx64x128x256 -> b_sx64x64x128
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1), # b_sx64x64x128-> b_sx64x32x64
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), # b_sx64x32x64 -> b_sx128x32x64
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2)), # b_sx128x32x64 -> b_sx64x16x32
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1), # b_sx128x16x32 -> b_sx128x8x16
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1), # b_sx128x8x16 -> b_sx256x8x16
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1), # b_sx256x8x16 -> b_sx512x4x8
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1), # b_sx256x4x8 -> b_sx512x4x8
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(4, 3), padding=0), # b_sx512x4x8 -> b_sx512x1x6
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

    def forward(self, input):
        # input = [b_s, 128, 256]
        input = input.unsqueeze(1)
        output = self.block_0(input)
        output = self.block_1(output)
        output = self.block_2(output)
        output = self.block_3(output)
        output = output.squeeze(2)
        # input = [b_s, 512, 6]
        return output

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bidirectional=True)
        self.embedding = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        # input = [src_len, b_s, input_size]
        output, _ = self.rnn(input)
        # output = [src_len, b_s, 2 * hidden_size]
        s_l, b_s, h_s = output.size()
        output = output.view(s_l * b_s, h_s)
        output = self.embedding(output)
        output = output.view(s_l, b_s, -1)
        # output = [src_len, b_s, output_size]
        return output

class CRNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(CRNN, self).__init__()
        self.cnn = CNN()
        self.rnn = LSTM(input_size, hidden_size, num_layers, output_size)

    def forward(self, input):
        # output = [b_s, 128, 256]
        output = self.cnn(input)
        # output = [b_s, input_size, src_len]
        output = output.permute(2, 0, 1)
        # output = [src_len, b_s, input_size]
        output = self.rnn(output)
        # output = [src_len, b_s, output_size]
        output = output.permute(1, 2, 0)
        # output = [b_s, src_len, output_size]
        return output