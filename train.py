import torch
from torch import nn
from torch.utils.data import DataLoader
from dataloader import WavPhnDataset, collate_fn_padd
from model import Segmentor
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

EPOCHS = 10
LEARNING_RATE = 0.001
val_ratio = 0.1


def train(model, train_data_loader, val_data_loader, loss_fn, optimiser, device, epochs, writer):
    step = 1
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        model.train()
        losses = []
        count = 0
        for input, target, len in train_data_loader:
            input, target = input.to(device), target.to(device)

            # calculate loss
            prediction = model(input).to(device)
            loss = loss_fn(prediction[0], target[0].long())
            losses.append(loss.item())

            # backpropagate error and update weights
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            count += 1

        val_test_acc, val_precision, val_recall, val_f1_score = test(model, val_data_loader, device, 'validation')
        epoch_loss = sum(losses) / count
        writer.add_scalar("validation accuracy", val_test_acc, step)
        writer.add_scalar("epoch loss", epoch_loss, step)
        step += 1
        writer.add_hparams(
            {"epoch": i},
            {
                "accuracy": val_test_acc,
                "loss": epoch_loss,
            },
        )
        print(f"loss: {epoch_loss}")
        print(f"validation accuracy: {val_test_acc}")
        print(f"validation precision: {val_precision}")
        print(f"validation recall: {val_recall}")
        print(f"validation f1-score: {val_f1_score}")
        print("---------------------------")
    print("Finished training")


def scores(prediction, target):
    acc = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    # for pred_i, targ_i in zip(prediction, target):
    #     if pred_i == targ_i:
    #         acc += 1
    for i in range(len(prediction)):
        if target[i] == 0:
            if prediction[i] == 0:
                acc += 1
                tn += 1
            else:
                if target[i-1] == 1 or target[i+1] == 1:
                    acc += 1
                    tn += 1
                else:
                    fp += 1
        if target[i] == 1:
            if prediction[i] == 1:
                acc += 1
                tp += 1
            else:
                if prediction[i-1] == 1 or prediction[i+1] == 1:
                    acc += 1
                    tp += 1
                else:
                    fn += 1

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = (2*precision*recall)/(precision+recall)

    return acc / len(target), precision, recall, f1_score


def test(model, data_loader, device, mode):
    model.eval()
    accuracy = 0
    precision = 0
    recall = 0
    f1_score = 0
    s = 0
    for input, target, len in data_loader:
        s += 1
        input, target = input.to(device), target.to(device)
        prediction = model(input).to(device)
        _, prediction_list = torch.max(prediction[0], 1)
        acc, pre, rec, f1 = scores(prediction_list, target[0])
        accuracy += acc
        precision += pre
        recall += rec
        f1_score += f1

        if mode == 'test':
            print(target[0])
            print("-----------------------------------------------------------")
            print(prediction_list)
            print("accuracy for this audio: " + str(acc))
            print("precision for this audio: " + str(pre))
            print("recall for this audio: " + str(rec))
            print("f1-score for this audio: " + str(f1))

    accuracy = accuracy / s
    precision = precision / s
    recall = recall / s
    f1_score = f1_score / s

    model.train()
    return accuracy, precision, recall, f1_score


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    dataset = WavPhnDataset("timit/train")
    train_len = len(dataset)
    train_split = int(train_len * (1 - val_ratio))
    val_split = train_len - train_split
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_split, val_split])
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    # construct model and assign it to device
    cnn = Segmentor().to(device)
    print(cnn)

    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.2, 0.8]).to(device))
    optimizer = {'adam': torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE),
                 'sgd': torch.optim.SGD(cnn.parameters(), lr=LEARNING_RATE, momentum=0.9)}

    writer = SummaryWriter(
        f"runs/epochAnalysis"
    )

    # train model
    train(cnn, train_dataloader, val_dataloader, loss_fn, optimizer['adam'], device, EPOCHS, writer)

    # save model
    torch.save(cnn.state_dict(), "segmentor.pth")
    print("Trained network saved at segmentor.pth")

    # load model
    # cnn = Segmentor().to(device)
    # cnn.load_state_dict(torch.load("segmentor.pth"))

    test_dataset = WavPhnDataset("timit/test")
    test_data_loader = DataLoader(test_dataset, batch_size=1)
    test_acc, test_precision, test_recall, test_f1_score = test(cnn, test_data_loader, device, 'test')
    print(f"test accuracy: {test_acc}")
    print(f"test precision: {test_precision}")
    print(f"test recall: {test_recall}")
    print(f"test f1-score: {test_f1_score}")
