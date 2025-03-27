from src import utils
import argparse
from src import dataset, metrics
from model import models
from torch.utils.data import DataLoader
import torch
from unlearn_strategies import strategies
import time

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
parser.add_argument("-save_model", type= bool, default= False, help= "Save trained model option")

# Unlearn configuration
parser.add_argument("-unlearn_method", type= str, default= "lipschitz",
                    choices= ["baseline",
                              "retrain",
                              "fine_tune",
                              "gradient_ascent",
                              "bad_teacher",
                              "scrub",
                              "amnesiac",
                              "boundary",
                              "ntk",
                              "fisher",
                              "unsir",
                              "ssd"],
                    help= "Baselines unlearn method")

parser.add_argument("-model_path", type= str,
                    help= "Trained model path")
parser.add_argument("-unlearn_class", type= int, help= "Class to unlearn")

# Training hyperparameter
parser.add_argument("-epochs", type= int, default= 1, help= "Training epochs")
parser.add_argument("-batch_size", type= int, default= 128, help= "Training batch size")
parser.add_argument("-lr", type=float, default= 1e-4, help='Learning rate')
parser.add_argument("-optimizer", type= str, default= "adam", choices= ["sgd, adam"])
parser.add_argument('-momentum', type=float, default= 0.5, help='SGD momentum (default: 0.5)')

# Unlearn Hyperparameter
parser.add_argument('-report_training', type= bool, default= True, help= "option to show training performance")
parser.add_argument('-report_interval', type= int, default= 5, help= "training performance report interval")

# Set seed
parser.add_argument("-seed", type=int,default= 0, help="Seed for runs")

args = parser.parse_args()


def main() -> None:
    # Set seed
    utils.set_seed(seed=args.seed)

    # Device
    device, device_name = utils.device_configuration(args=args)
    print(f"Unlearning scenario: {args.scenario} Dataset: {args.dataset} Unlearn method: {args.unlearn_method} Device: {device}")

    # Dataset
    train_dataset, test_dataset, num_classes, num_channels = dataset.get_dataset(
        dataset_name=args.dataset, root=args.root
    )

    retain_dataset, unlearn_dataset = dataset.split_unlearn_dataset(
        data_list=train_dataset,
        unlearn_class=args.unlearn_class
    )

    retain_loader = DataLoader(retain_dataset, batch_size=args.batch_size, shuffle=True)
    unlearn_loader = DataLoader(unlearn_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # Model preparation
    model = getattr(models, args.model)(
        num_classes=num_classes, input_channels=num_channels).to(device)
    unlearning_teacher = getattr(models, args.model)(
        num_classes=num_classes, input_channels=num_channels).to(device)

    if args.unlearn_method != "retrain":
        # Load trained model to unlearn
        model.load_state_dict(torch.load(args.model_path))

    start_time = time.time()
    # Unlearn
    unlearned_model = getattr(strategies, args.unlearn_method)(
        args=args,
        model=model,
        unlearning_teacher= unlearning_teacher,
        unlearn_class= args.unlearn_class,
        unlearn_loader=unlearn_loader,
        retain_loader=retain_loader,
        test_loader=test_loader,
        num_channels=num_channels,
        num_classes=num_classes,
        device=device
    )
    end_time = time.time()
    runtime = end_time - start_time

    # Evaluation after unlearning
    retain_acc = metrics.evaluate(val_loader=retain_loader, model=unlearned_model, device=device)['Acc']
    unlearn_acc = metrics.evaluate(val_loader=unlearn_loader, model=unlearned_model, device=device)['Acc']
    mia = metrics.mia(
        retain_loader=retain_loader,
        forget_loader=unlearn_loader,
        test_loader=test_loader,
        model=unlearned_model)
    print(f"Unlearned - Retain acc: {retain_acc} Unlearn_acc: {unlearn_acc} MIA: {mia} Runtime: {runtime}s")

    if args.save_model:
        utils.save_model(
            model_arc=args.model,
            model=unlearned_model,
            scenario=args.scenario,
            model_name=args.unlearn_method,
            model_root=args.model_root,
            dataset_name=args.dataset,
            train_acc=retain_acc,
            test_acc=unlearn_acc
        )

if __name__ == "__main__":
    main()