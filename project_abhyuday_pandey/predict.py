# predict.py

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from PIL import Image

from model import MyCustomModel
from dataset import get_dataloader, transform


#  GRADER FUNCTION (must work independently)
def predict_images(list_of_paths):
    model = MyCustomModel()
    model.load_state_dict(torch.load("checkpoints/best_model.pth"))
    model.eval()

    images = []

    for path in list_of_paths:
        img = Image.open(path)
        img = transform(img)
        images.append(img)

    images = torch.stack(images)

    with torch.no_grad():
        logits = model(images)
        probs = torch.sigmoid(logits)

    labels = ["HiPt" if p > 0.5 else "LoPt" for p in probs]

    return labels


#  LOCAL TESTING (ROC curve)
def evaluate_model():
    model = MyCustomModel()
    model.load_state_dict(torch.load("checkpoints/best_model.pth"))
    model.eval()

    test_loader = get_dataloader(train=False)

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images)
            probs = torch.sigmoid(logits)

            all_probs.extend(probs.squeeze().tolist())
            all_labels.extend(labels.tolist())

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    print(f"AUC: {roc_auc:.4f}")

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("roc_curve.png")
    plt.close()


if __name__ == "__main__":
    evaluate_model()
