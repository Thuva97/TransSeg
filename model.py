from torch import nn
import torch
from torch.utils.data import DataLoader

from dataloader import WavPhnDataset, collate_fn_padd


class Segmentor(nn.Module):
    def __init__(self):
        super(Segmentor, self).__init__()

        self.encoderLayer = nn.TransformerEncoderLayer(
                                              d_model=43,
                                              nhead=1,
                                              dim_feedforward=2048,
                                              dropout=0.3,
                                              activation="relu")
        self.tnn = nn.TransformerEncoder(self.encoderLayer, num_layers=2)

        self.rnn = nn.LSTM(43,
                           100,
                           num_layers=2,
                           batch_first=True,
                           dropout=0.3,
                           bidirectional=True)

        self.bin_classifier = nn.Sequential(
            nn.PReLU(),
            nn.Linear(200, 2 * 39),
            nn.PReLU(),
            nn.Linear(2 * 39, 2)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.tnn(x)
        x = self.rnn(x)
        x = self.bin_classifier(x[0])
        return x


if __name__ == "__main__":
    cnn = Segmentor().to("cuda")
    dataset = WavPhnDataset("timit/train")
    trainDataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn_padd)
    for e in trainDataloader:
        print(e)
        e1 = e[0].to("cuda")
        result = cnn(e1)
        print(result)
        print(result.size())
        break

    loss = nn.CrossEntropyLoss()
    input = torch.Tensor([[0.4862, 0.5066],
        [0.4853, 0.5069],
        [0.4847, 0.5059],
        [0.4836, 0.5045],
        [0.4829, 0.5059]]).to("cuda")
    target = torch.Tensor([1, 1, 1, 1, 1]).long().to("cuda")
    output = loss(input, target)
    print(input)
    print(list(target))
    print(output)


