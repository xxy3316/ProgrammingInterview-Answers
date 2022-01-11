import torch

from readData import test_dataloader
from matplotlib import pyplot as plt

def test(model):



    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    predict_ls = [0.2]
    true_ls = [0.2]
    with torch.no_grad():

        for images,targets in test_dataloader:
            images = images.to(device)
            targets = targets.to(device)
            if len(images) != 100:
                continue
            y_logits = model(images)
            targets = float(sum(targets)) / float(len(targets))
            predict_ls.append(y_logits)
            true_ls.append(targets)

    predict_ls.append(0.8)
    true_ls.append(0.8)
    plt.title("Prediction Result")
    plt.xlabel("MIL predictions")
    plt.ylabel("True")
    plt.plot(predict_ls,true_ls,"ob")
    plt.show()
