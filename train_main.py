import copy
from src import dataset, metrics, utils
import argparse
import torch
from torch.utils.data import DataLoader
from model import models
from torch import nn
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
# Device
parser.add_argument("-gpu", type= bool, default= True, help= "use gpu or not")
# Dataset
parser.add_argument("-root", type= str, default= "./data", help= "Dataset root directory")
parser.add_argument("-dataset", type= str, help= "Dataset configuration",
                    choices=["MNist",
                             "FMNist",
                             "Cifar10",
                             "Cifar100",
                             "TinyImagenet"])
# Model
parser.add_argument("-model_root", type= str, default= "./checkpoint", help= "Dataset root directory")
parser.add_argument("-model", type= str, default= "ResNet18", help= "Model selection")
parser.add_argument("-save_model", type= bool, default= True, help= "Save trained model option")

# Training hyperparameter
parser.add_argument("-epochs", type= int, default= 30, help= "Training epochs")
parser.add_argument("-batch_size", type= int, default= 128, help= "Training batch size")
parser.add_argument("-lr", type=float, default= 1e-4, help='Learning rate')
parser.add_argument("-optimizer", type= str, default= "adam", choices= ["sgd, adam"])
parser.add_argument('-momentum', type=float, default= 0.5, help='SGD momentum (default: 0.5)')
parser.add_argument("-scenario", type= str, default= "class",
                    choices= ["class", "client", "sample"], help= "Training and unlearning scenario")

parser.add_argument('-report_training', type= bool, default= False, help= "option to show training performance")
parser.add_argument('-report_interval', type= int, default= 5, help= "training performance report interval")

# Set seed
parser.add_argument("-seed", type=int,default= 0, help="Seed for runs")

args = parser.parse_args()

if __name__ == "__main__":
    # Set seed
    utils.set_seed(seed= args.seed)

    # Device
    device, device_name = utils.device_configuration(args= args)

    # Dataset
    train_dataset, test_dataset, num_classes, num_channels = dataset.get_dataset(
        dataset_name= args.dataset, root= args.root
    )

    train_loader = DataLoader(train_dataset, batch_size= args.batch_size, shuffle= True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle= True)

    # Model preparation
    model = getattr(models, args.model)(
        num_classes= num_classes, input_channels= num_channels).to(device)

    if args.optimizer not in ["sgd", "adam"]:
        raise Exception("select correct optimizer")
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    loss_func = nn.CrossEntropyLoss().to(device)
    max_test_acc = 0.0
    max_train_acc = 0.0

    for epoch in tqdm(range(1, args.epochs + 1)):
        loss_list = []
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.long().to(device)

            model.zero_grad()
            output = model(images)
            loss = loss_func(output, labels)
            loss.backward()
            optimizer.step()

            # Evaluation preparation
            loss_list.append(loss.item())

        mean_loss = np.mean(np.array(loss_list))
        train_acc = metrics.evaluate(val_loader= train_loader, model= model, device= device)['Acc']
        test_acc = metrics.evaluate(val_loader= test_loader, model= model, device= device)['Acc']
        if args.report_training:
            tqdm.write(f"Epochs: {epoch} Train Loss: {mean_loss:.4f} Train Acc: {train_acc} Test Acc: {test_acc}")

        if test_acc >= max_test_acc:
            max_test_acc = test_acc
            max_train_acc = train_acc
            best_model = copy.deepcopy(model)

    utils.save_model(
        model_arc=args.model,
        model=best_model,
        scenario=args.scenario,
        model_name="baseline",
        model_root=args.model_root,
        dataset_name=args.dataset,
        train_acc=max_train_acc,
        test_acc=max_test_acc
    )