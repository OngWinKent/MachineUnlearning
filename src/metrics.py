"""
Evaluation metrics
"""
from torch.nn import functional as F
import copy
import os
import torch
from typing import Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Classification accuracy metrics
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds)) * 100


def validation_step(model, batch, device):
    images, clabels = batch
    images, clabels = images.to(device), clabels.long().to(device)
    out = model(images)  # Generate predictions
    loss = F.cross_entropy(out, clabels)  # Calculate loss
    acc = accuracy(out, clabels)  # Calculate accuracy
    return {"Loss": loss.detach(), "Acc": acc}


def validation_epoch_end(model, outputs):
    batch_losses = [x["Loss"] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
    batch_accs = [x["Acc"] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
    return {"Loss": round(epoch_loss.item(), 4), "Acc": round(epoch_acc.item(), 4)}


@torch.no_grad()
def evaluate(model, val_loader, device):
    copy_model= copy.deepcopy(model)
    copy_model.eval()
    outputs = [validation_step(copy_model, batch, device) for batch in val_loader]
    return validation_epoch_end(copy_model, outputs)


def collect_entropy(
    data_loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device
) -> np.array:
    prob = collect_prob(data_loader, model)
    enp = entropy(prob).cpu().detach().numpy()
    return enp


def plot_entropy(
    data_loader: DataLoader,
    original_model: torch.nn.Module,
    unlearn_model: torch.nn.Module,
    device: torch.device
) -> None:
    original_enp = collect_entropy(data_loader= data_loader, model= original_model, device= device)
    unlearn_enp = collect_entropy(data_loader= data_loader, model= unlearn_model, device= device)

    print(f"Original model mean entropy: {np.mean(original_enp)}")
    print(f"Unlearn model mean entropy: {np.mean(unlearn_enp)}")
    # Plot the entropy
    plt.figure(figsize=(10, 6))
    #plt.hist(np.log(original_enp), bins=50, alpha=0.7, label= "Original")
    #plt.hist(np.log(unlearn_enp), bins=50, alpha=0.7, label= "Unlearn")
    plt.hist(original_enp, bins=50, alpha=0.7, label= "Original")
    plt.hist(unlearn_enp, bins=50, alpha=0.7, label= "Unlearn")
    plt.legend()
    plt.title('Entropy Distribution of Model Predictions')
    plt.xlabel('Entropy')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


# Reference of Membership Inference Attack(Code Implementation): https://github.com/if-loops/selective-synaptic-dampening
def entropy(
    p: torch.Tensor, # softmax input
    dim: int = -1,
    keepdim: bool = False
) -> torch.Tensor:

    # Ensure p is softmax output
    #enp = -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)

    # Compute entropy: H(p) = -sum(p * log(p))
    epsilon = 1e-10 # Add a small value avoid log 0
    enp = -torch.sum(p * torch.log(p + epsilon), dim=-1)

    return enp


def collect_prob(
    data_loader,
    model
):

    data_loader = DataLoader(
        data_loader.dataset, batch_size=1, shuffle=False
    )
    prob = []
    with torch.no_grad():
        #for batch in data_loader:
        for batch in tqdm(data_loader):
            batch = [tensor.to(next(model.parameters()).device) for tensor in batch]
            data, target = batch
            output = model(data)
            prob.append(F.softmax(output, dim=-1).data)
    return torch.cat(prob)


# https://arxiv.org/abs/2205.08096
def get_membership_attack_data(
    retain_loader,
    forget_loader,
    test_loader,
    model
):
    retain_prob = collect_prob(retain_loader, model)
    forget_prob = collect_prob(forget_loader, model)
    test_prob = collect_prob(test_loader, model)

    forget_enp = np.mean(entropy(forget_prob).cpu().detach().numpy()).item()

    X_r = (
        torch.cat([entropy(retain_prob), entropy(test_prob)])
        .cpu()
        .numpy()
        .reshape(-1, 1)
    )
    Y_r = np.concatenate([np.ones(len(retain_prob)), np.zeros(len(test_prob))])

    X_f = entropy(forget_prob).cpu().numpy().reshape(-1, 1)
    Y_f = np.concatenate([np.ones(len(forget_prob))])
    return X_f, Y_f, X_r, Y_r, forget_enp


# https://arxiv.org/abs/2205.08096
def mia(
    retain_loader,
    forget_loader,
    test_loader,
    model
) -> float:
    copy_model = copy.deepcopy(model) # avoid overwriting
    X_f, Y_f, X_r, Y_r, forget_enp = get_membership_attack_data(
        retain_loader, forget_loader, test_loader, copy_model
    )
    # clf = SVC(C=3,gamma='auto',kernel='rbf')
    clf = LogisticRegression(
        class_weight="balanced", solver="lbfgs", multi_class="multinomial"
    )
    clf.fit(X_r, Y_r)
    results = clf.predict(X_f)
    return round(results.mean() * 100, 4)


def model_evaluation(
    retain_loader: DataLoader,
    unlearn_loader: DataLoader,
    test_loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device
)-> Tuple[float, float, float]:

    retain_acc = evaluate(val_loader= retain_loader, model= model, device= device)["Acc"]
    unlearn_acc = evaluate(val_loader= unlearn_loader, model= model, device= device)["Acc"]
    mia_asr = mia(retain_loader= retain_loader, forget_loader= unlearn_loader, test_loader= test_loader, model= model)

    return retain_acc, unlearn_acc, mia_asr