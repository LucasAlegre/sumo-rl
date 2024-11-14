import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # In this example, we don't change the model architecture
        # due to simplicity.
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        self.fc = nn.Linear(192, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = x.view(-1, 192)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


input_image = torch.tensor([[[[1, 2, 3, 4, 5],
                              [6, 7, 8, 9, 10],
                              [11, 12, 13, 14, 15],
                              [16, 17, 18, 19, 20],
                              [21, 22, 23, 24, 25]]]], dtype=torch.float32)

model = ConvNet()
output_image = model(input_image)

print("\ninput_image.shape", input_image.shape)
print("input_image\n", input_image)
print("\noutput_image.shape", output_image.shape)
print("output_image\n", output_image)
