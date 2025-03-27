"""
Unlearning strategies file
"""
import copy
import torch
from torch.utils.data import DataLoader, ConcatDataset
import argparse
from copy import deepcopy
import random
from unlearn_strategies import utils, unlearn
from unlearn_strategies.unlearn import FGSM, ParameterPerturber, DistillKL, adjust_learning_rate, train_distill
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import torch.nn.functional as F
from torch import nn
from src import metrics


def baseline(
    args: argparse.Namespace,
    model: torch.nn.Module,
    unlearning_teacher: torch.nn.Module,
    unlearn_class: int,
    unlearn_loader: DataLoader,
    retain_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    num_channels: int,
    device: torch.device
) -> torch.nn.Module:
    return model


def retrain(
    args: argparse.Namespace,
    model: torch.nn.Module,
    unlearning_teacher: torch.nn.Module,
    unlearn_class: int,
    unlearn_loader: DataLoader,
    retain_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    num_channels: int,
    device: torch.device
) -> torch.nn.Module:

    # Retrain model from scratch without unlearn dataset
    retrain_model = utils.training_optimization(
        model= model,
        train_loader= retain_loader,
        test_loader= test_loader,
        epochs= 30,
        device= device,
        desc= "Retraining model"
    )

    return retrain_model


def fine_tune(
    args: argparse.Namespace,
    model: torch.nn.Module,
    unlearning_teacher: torch.nn.Module,
    unlearn_class: int,
    unlearn_loader: DataLoader,
    retain_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    num_channels: int,
    device: torch.device,
) -> torch.nn.Module:

    # Fine tune model with retain dataset
    ft_model = utils.training_optimization(
        model= model,
        train_loader= retain_loader,
        test_loader= test_loader,
        epochs= 5,
        device= device,
        desc= "Fine-tuning model"
    )

    return ft_model


# Unrolling SGD: https://github.com/cleverhans-lab/unrolling-sgd
def gradient_ascent(
    args: argparse.Namespace,
    model: torch.nn.Module,
    unlearning_teacher: torch.nn.Module,
    unlearn_class: int,
    unlearn_loader: DataLoader,
    retain_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    num_channels: int,
    device: torch.device,
) -> torch.nn.Module:

    epochs = 5
    unlearned_model = copy.deepcopy(model)
    optimizer = torch.optim.SGD(unlearned_model.parameters(), lr=1e-4, momentum= 0.5)
    loss_func = nn.CrossEntropyLoss().to(device)

    for epoch in tqdm(range(1, epochs + 1), desc= "Gradient ascent unlearning"):
        loss_list = []
        for images, labels in unlearn_loader:
            images, labels= images.to(device), labels.long().to(device)
            unlearned_model.zero_grad()
            output = unlearned_model(images)
            # gradient ascent loss
            loss = 1 - loss_func(output, labels)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        mean_loss = np.mean(np.array(loss_list))
        train_acc = metrics.evaluate(val_loader=retain_loader, model=unlearned_model, device=device)['Acc']
        test_acc = metrics.evaluate(val_loader=test_loader, model=unlearned_model, device=device)['Acc']
        #tqdm.write(f"Epochs: {epoch} Train Loss: {mean_loss:.4f} Train Acc: {train_acc} Test acc: {test_acc}")

    return unlearned_model


# Bad Teacher: https://github.com/vikram2000b/bad-teaching-unlearning
def bad_teacher(
    args: argparse.Namespace,
    model: torch.nn.Module,
    unlearning_teacher: torch.nn.Module,
    unlearn_class: int,
    unlearn_loader: DataLoader,
    retain_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    num_channels: int,
    device: torch.device,
) -> torch.nn.Module:

    student_model = deepcopy(model)
    KL_temperature = 1
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.0001)
    retain_train_subset = random.sample(
        retain_loader.dataset, int(0.3 * len(retain_loader.dataset)))

    unlearn.blindspot_unlearner(
        model=student_model,
        unlearning_teacher=unlearning_teacher,
        full_trained_teacher=model,
        retain_data=retain_train_subset,
        forget_data=unlearn_loader.dataset,
        epochs=1,
        optimizer=optimizer,
        lr=0.0001,
        batch_size=256,
        device=device,
        KL_temperature=KL_temperature,
    )

    return student_model


# SCRUB: https://github.com/meghdadk/SCRUB/tree/main
def scrub(
    args: argparse.Namespace,
    model: torch.nn.Module,
    unlearning_teacher: torch.nn.Module,
    unlearn_class: int,
    unlearn_loader: DataLoader,
    retain_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    num_channels: int,
    device: torch.device,
) -> torch.nn.Module:

    # Parameters
    optim = 'sgd'
    gamma = 0.99
    alpha = 0.001
    beta = 0
    smoothing = 0.0
    msteps = 2
    clip = 0.2
    sstart = 10
    kd_T = 4
    distill = 'kd'

    sgda_batch_size = 128
    del_batch_size = 32
    sgda_epochs = 3
    sgda_learning_rate = 0.0005
    lr_decay_epochs = [3, 5, 9]
    lr_decay_rate = 0.1
    sgda_weight_decay = 5e-4
    sgda_momentum = 0.9

    # Deep copy avoid overwriting
    model_t = copy.deepcopy(unlearning_teacher)
    model_s = copy.deepcopy(model)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(kd_T)
    criterion_kd = DistillKL(kd_T)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)  # other knowledge distillation loss

    optimizer = torch.optim.SGD(
        trainable_list.parameters(),
        lr= sgda_learning_rate,
        momentum= sgda_momentum,
        weight_decay= sgda_weight_decay)

    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    for epoch in tqdm(range(1, sgda_epochs + 1), desc= "SCRUB Unlearning"):

        lr = adjust_learning_rate(
            epoch= epoch,
            optimizer= optimizer,
            lr_decay_epochs= lr_decay_epochs,
            sgda_learning_rate= sgda_learning_rate,
            lr_decay_rate= lr_decay_rate
        )

        maximize_loss = 0
        if epoch <= msteps:
            maximize_loss = train_distill(
                epoch= epoch,
                train_loader= unlearn_loader,
                module_list= module_list,
                swa_model= None,
                criterion_list= criterion_list,
                optimizer= optimizer,
                gamma= gamma,
                alpha= alpha,
                beta= beta,
                split= "maximize")
        train_acc, train_loss = train_distill(
            epoch= epoch,
            train_loader= retain_loader,
            module_list= module_list,
            swa_model= None,
            criterion_list= criterion_list,
            optimizer= optimizer,
            gamma= gamma,
            alpha= alpha,
            beta= beta,
            split= "minimize",
            quiet= True)

    return model_s


# Amnesiac Unlearning: https://github.com/lmgraves/AmnesiacML
def amnesiac(
    args: argparse.Namespace,
    model: torch.nn.Module,
    unlearning_teacher: torch.nn.Module,
    unlearn_class: int,
    unlearn_loader: DataLoader,
    retain_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    num_channels: int,
    device: torch.device,
)-> torch.nn.Module:
    
    unlearninglabels = list(range(num_classes))
    unlearning_trainset = []

    unlearninglabels.remove(unlearn_class)

    for x, clabel in unlearn_loader.dataset:
        unlearning_trainset.append((x, random.choice(unlearninglabels)))

    for x, y in retain_loader.dataset:
        unlearning_trainset.append((x, y))

    unlearning_train_set_dl = DataLoader(
        unlearning_trainset, 128, pin_memory=True, shuffle=True
    )

    unlearned_model = utils.training_optimization(
        model= model, 
        train_loader= unlearning_train_set_dl,
        test_loader= test_loader,
        epochs= 5,
        device= device,
        desc= "Amnesiac unlearning")
    
    return unlearned_model


# Boundary Unlearning: https://github.com/TY-LEE-KR/Boundary-Unlearning-Code
def boundary(
    args: argparse.Namespace,
    model: torch.nn.Module,
    unlearning_teacher: torch.nn.Module,
    unlearn_class: int,
    unlearn_loader: DataLoader,
    retain_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    num_channels: int,
    device: torch.device
) -> torch.nn.Module:
    # Boundary Shrink
    # Hyperparameter
    bound = 0.1
    poison_epoch = 10
    extra_exp = None
    lambda_ = 0.7
    bias = -0.5
    slope = 5.0

    #norm = True  # None #True if data_name != "mnist" else False
    if num_channels == 3:
        norm = True
    else:
        norm = False
    random_start = False  # False if attack != "pgd" else True

    test_model = deepcopy(model).to(device)
    unlearn_model = deepcopy(model).to(device)
    # adv = LinfPGD(test_model, bound, step, iter, norm, random_start, device)
    adv = FGSM(test_model, bound, norm, random_start, device)
    forget_data_gen = unlearn.inf_generator(unlearn_loader)
    batches_per_epoch = len(unlearn_loader)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=0.00001, momentum=0.9)

    num_hits = 0
    num_sum = 0
    nearest_label = []

    for itr in tqdm(range(poison_epoch * batches_per_epoch)):

        x, y = forget_data_gen.__next__()
        x = x.to(device)
        y = y.to(device)
        test_model.eval()
        x_adv = adv.perturb(x, y, target_y=None, model=test_model, device=device)
        adv_logits = test_model(x_adv)
        pred_label = torch.argmax(adv_logits, dim=1)
        if itr >= (poison_epoch - 1) * batches_per_epoch:
            nearest_label.append(pred_label.tolist())
        num_hits += (y != pred_label).float().sum()
        num_sum += y.shape[0]

        # adv_train
        unlearn_model.train()
        unlearn_model.zero_grad()
        optimizer.zero_grad()

        ori_logits = unlearn_model(x)
        ori_loss = criterion(ori_logits, pred_label)

        # loss = ori_loss  # - KL_div
        if extra_exp == 'curv':
            ori_curv = unlearn.curvature(model, x, y, h=0.9)[1]
            cur_curv = unlearn.curvature(unlearn_model, x, y, h=0.9)[1]
            delta_curv = torch.norm(ori_curv - cur_curv, p=2)
            loss = ori_loss + lambda_ * delta_curv  # - KL_div
        elif extra_exp == 'weight_assign':
            weight = unlearn.weight_assign(adv_logits, pred_label, bias=bias, slope=slope)
            ori_loss = (torch.nn.functional.cross_entropy(ori_logits, pred_label, reduction='none') * weight).mean()
            loss = ori_loss
        else:
            loss = ori_loss  # - KL_div
        loss.backward()
        optimizer.step()
    return unlearn_model


# Neural Tangent Kernel: https://github.com/AdityaGolatkar/SelectiveForgetting
def ntk(
    args: argparse.Namespace,
    model: torch.nn.Module,
    unlearning_teacher: torch.nn.Module,
    unlearn_class: int,
    unlearn_loader: DataLoader,
    retain_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    num_channels: int,
    device: torch.device,
) -> torch.nn.Module:
    
    def delta_w_utils(model_init, dataloader, name="complete"):
        model_init.eval()
        dataloader = torch.utils.data.DataLoader(
            dataloader.dataset, batch_size=1, shuffle=False
        )
        G_list = []
        f0_minus_y = []
        for idx, batch in enumerate(
            tqdm(dataloader)
        ):  # (tqdm(dataloader,leave=False)):
            batch = [
                tensor.to(next(model_init.parameters()).device) for tensor in batch
            ]
            input, target = batch

            target = target.cpu().detach().numpy()
            output = model_init(input)
            G_sample = []
            for cls in range(num_classes):
                grads = torch.autograd.grad(
                    output[0, cls], model_init.parameters(), retain_graph=True
                )
                grads = np.concatenate([g.view(-1).cpu().numpy() for g in grads])
                G_sample.append(grads)
                G_list.append(grads)
                p = (
                    torch.nn.functional.softmax(output, dim=1)
                    .cpu()
                    .detach()
                    .numpy()
                    .transpose()
                )
                p[target] -= 1
                f0_y_update = deepcopy(p)
            f0_minus_y.append(f0_y_update)
        return np.stack(G_list).transpose(), np.vstack(f0_minus_y)

    #############################################################################################
    model_init = deepcopy(model)
    G_r, f0_minus_y_r = delta_w_utils(deepcopy(model), retain_loader, "complete")
    print("GOT GR")
    # np.save('NTK_data/G_r.npy',G_r)
    # np.save('NTK_data/f0_minus_y_r.npy',f0_minus_y_r)
    # del G_r, f0_minus_y_r

    G_f, f0_minus_y_f = delta_w_utils(deepcopy(model), unlearn_loader, "retain")
    print("GOT GF")
    # np.save('NTK_data/G_f.npy',G_f)
    # np.save('NTK_data/f0_minus_y_f.npy',f0_minus_y_f)
    # del G_f, f0_minus_y_f

    # G_r = np.load('NTK_data/G_r.npy')
    # G_f = np.load('NTK_data/G_f.npy')
    G = np.concatenate([G_r, G_f], axis=1)
    print("GOT G")
    # np.save('NTK_data/G.npy',G)
    # del G, G_f, G_r

    # f0_minus_y_r = np.load('NTK_data/f0_minus_y_r.npy')
    # f0_minus_y_f = np.load('NTK_data/f0_minus_y_f.npy')
    f0_minus_y = np.concatenate([f0_minus_y_r, f0_minus_y_f])

    # np.save('NTK_data/f0_minus_y.npy',f0_minus_y)
    # del f0_minus_y, f0_minus_y_r, f0_minus_y_f

    weight_decay = 0.1

    # G = np.load('NTK_data/G.npy')
    theta = G.transpose().dot(G) + (
        len(retain_loader.dataset) + len(unlearn_loader.dataset)
    ) * weight_decay * np.eye(G.shape[1])
    # del G

    theta_inv = np.linalg.inv(theta)

    # np.save('NTK_data/theta.npy',theta)
    # del theta

    # G = np.load('NTK_data/G.npy')
    # f0_minus_y = np.load('NTK_data/f0_minus_y.npy')
    w_complete = -G.dot(theta_inv.dot(f0_minus_y))

    # np.save('NTK_data/theta_inv.npy',theta_inv)
    # np.save('NTK_data/w_complete.npy',w_complete)
    # del G, f0_minus_y, theta_inv, w_complete

    # G_r = np.load('NTK_data/G_r.npy')
    num_to_retain = len(retain_loader.dataset)
    theta_r = G_r.transpose().dot(G_r) + num_to_retain * weight_decay * np.eye(
        G_r.shape[1]
    )
    # del G_r

    theta_r_inv = np.linalg.inv(theta_r)
    # np.save('NTK_data/theta_r.npy',theta_r)
    # del theta_r

    # G_r = np.load('NTK_data/G_r.npy')
    # f0_minus_y_r = np.load('NTK_data/f0_minus_y_r.npy')
    w_retain = -G_r.dot(theta_r_inv.dot(f0_minus_y_r))

    # np.save('NTK_data/theta_r_inv.npy',theta_r_inv)
    # np.save('NTK_data/w_retain.npy',w_retain)
    # del G_r, f0_minus_y_r, theta_r_inv, w_retain

    def get_delta_w_dict(delta_w, model):
        # Give normalized delta_w
        delta_w_dict = OrderedDict()
        params_visited = 0
        for k, p in model.named_parameters():
            num_params = np.prod(list(p.shape))
            update_params = delta_w[params_visited : params_visited + num_params]
            delta_w_dict[k] = torch.Tensor(update_params).view_as(p)
            params_visited += num_params
        return delta_w_dict

    #### Scrubbing Direction
    # w_complete = np.load('NTK_data/w_complete.npy')
    # w_retain = np.load('NTK_data/w_retain.npy')
    print("got prelims, calculating delta_w")
    delta_w = (w_retain - w_complete).squeeze()
    print("got delta_w")
    # delta_w_copy = deepcopy(delta_w)
    # delta_w_actual = vectorize_params(model0)-vectorize_params(model)

    # print(f'Actual Norm-: {np.linalg.norm(delta_w_actual)}')
    # print(f'Predtn Norm-: {np.linalg.norm(delta_w)}')
    # scale_ratio = np.linalg.norm(delta_w_actual)/np.linalg.norm(delta_w)
    # print('Actual Scale: {}'.format(scale_ratio))
    # log_dict['actual_scale_ratio']=scale_ratio
    def vectorize_params(model):
        param = []
        for p in model.parameters():
            param.append(p.data.view(-1).cpu().numpy())
        return np.concatenate(param)

    m_pred_error = (
        vectorize_params(model) - vectorize_params(model_init) - w_retain.squeeze()
    )
    print(f"Delta w -------: {np.linalg.norm(delta_w)}")

    inner = np.inner(
        delta_w / np.linalg.norm(delta_w), m_pred_error / np.linalg.norm(m_pred_error)
    )
    print(f"Inner Product--: {inner}")

    if inner < 0:
        angle = np.arccos(inner) - np.pi / 2
        print(f"Angle----------:  {angle}")

        predicted_norm = np.linalg.norm(delta_w) + 2 * np.sin(angle) * np.linalg.norm(
            m_pred_error
        )
        print(f"Pred Act Norm--:  {predicted_norm}")
    else:
        angle = np.arccos(inner)
        print(f"Angle----------:  {angle}")

        predicted_norm = np.linalg.norm(delta_w) + 2 * np.cos(angle) * np.linalg.norm(
            m_pred_error
        )
        print(f"Pred Act Norm--:  {predicted_norm}")

    predicted_scale = predicted_norm / np.linalg.norm(delta_w)
    predicted_scale
    print(f"Predicted Scale:  {predicted_scale}")
    # log_dict['predicted_scale_ratio']=predicted_scale

    # def NIP(v1,v2):
    #     nip = (np.inner(v1/np.linalg.norm(v1),v2/np.linalg.norm(v2)))
    #     print(nip)
    #     return nip
    # nip=NIP(delta_w_actual,delta_w)
    # log_dict['nip']=nip
    scale = predicted_scale
    direction = get_delta_w_dict(delta_w, model)

    for k, p in model.named_parameters():
        p.data += (direction[k] * scale).to(device)

    return model


# Fisher Forgertting: https://github.com/AdityaGolatkar/SelectiveForgetting
def fisher(
    args: argparse.Namespace,
    model: torch.nn.Module,
    unlearning_teacher: torch.nn.Module,
    unlearn_class: int,
    unlearn_loader: DataLoader,
    retain_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    num_channels: int,
    device: torch.device,
) -> torch.nn.Module:
    
    def hessian(dataset, model):
        model.eval()
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        loss_fn = nn.CrossEntropyLoss()

        for p in model.parameters():
            p.grad_acc = 0
            p.grad2_acc = 0

        for data, orig_target in tqdm(train_loader):
            data, orig_target = data.to(device), orig_target.to(device)
            output = model(data)
            prob = F.softmax(output, dim=-1).data

            for y in range(output.shape[1]):
                target = torch.empty_like(orig_target).fill_(y)
                loss = loss_fn(output, target)
                model.zero_grad()
                loss.backward(retain_graph=True)
                for p in model.parameters():
                    if p.requires_grad:
                        p.grad_acc += (orig_target == target).float() * p.grad.data
                        p.grad2_acc += prob[:, y] * p.grad.data.pow(2)

        for p in model.parameters():
            p.grad_acc /= len(train_loader)
            p.grad2_acc /= len(train_loader)

    def get_mean_var(p, forget_class, is_base_dist=False, alpha=3e-6):
        var = deepcopy(1.0 / (p.grad2_acc + 1e-8))
        var = var.clamp(max=1e3)
        if p.size(0) == num_classes:
            var = var.clamp(max=1e2)
        var = alpha * var

        if p.ndim > 1:
            var = var.mean(dim=1, keepdim=True).expand_as(p).clone()
        if not is_base_dist:
            mu = deepcopy(p.data0.clone())
        else:
            mu = deepcopy(p.data0.clone())
        if p.size(0) == num_classes:
            mu[forget_class] = 0
            var[forget_class] = 0.0001
        if p.size(0) == num_classes:
            # Last layer
            var *= 10
        elif p.ndim == 1:
            # BatchNorm
            var *= 10
        #         var*=1
        return mu, var

    for p in model.parameters():
        p.data0 = deepcopy(p.data.clone())

    hessian(retain_loader.dataset, model)

    fisher_dir = []
    alpha = 1e-6
    for i, p in enumerate(model.parameters()):
        mu, var = get_mean_var(p, unlearn_class, False, alpha=alpha)
        p.data = mu + var.sqrt() * torch.empty_like(p.data0).normal_()
        fisher_dir.append(var.sqrt().view(-1).cpu().detach().numpy())

    return model


# Selective Impair and Repair: https://github.com/vikram2000b/Fast-Machine-Unlearning
def unsir(
    args: argparse.Namespace,
    model: torch.nn.Module,
    unlearning_teacher: torch.nn.Module,
    unlearn_class: int,
    unlearn_loader: DataLoader,
    retain_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    num_channels: int,
    device: torch.device,
) -> torch.nn.Module:
    
    classwise_train = unlearn.get_classwise_ds(
        ConcatDataset((retain_loader.dataset, unlearn_loader.dataset)), num_classes
    )
    noise_batch_size = 32
    retain_valid_dl = DataLoader(retain_loader.dataset, batch_size=noise_batch_size)
    # collect some samples from each class
    num_samples = 500
    retain_samples = []
    for i in range(num_classes):
        if i != unlearn_class:
            retain_samples += classwise_train[i][:num_samples]

    forget_class_label = unlearn_class
    img_shape = next(iter(retain_loader.dataset))[0].shape[-1]
    noise = unlearn.UNSIR_noise(noise_batch_size, num_channels, img_shape, img_shape).to(device)
    noise = unlearn.UNSIR_noise_train(
        noise, model, forget_class_label, 25, noise_batch_size, device=device
    )
    noisy_loader = unlearn.UNSIR_create_noisy_loader(
        noise,
        forget_class_label,
        retain_samples,
        batch_size=noise_batch_size,
        device=device
    )
    # impair step
    model = utils.training_optimization(
        model= model, 
        epochs= 1,
        train_loader= noisy_loader, 
        test_loader= retain_valid_dl,
        opt= "adam",
        device=device,
        desc= "UNSIR impair step"
    )
    # repair step
    other_samples = []
    for i in range(len(retain_samples)):
        other_samples.append(
            (
                retain_samples[i][0].cpu(),
                torch.tensor(retain_samples[i][1])
            )
        )

    heal_loader = torch.utils.data.DataLoader(
        other_samples, batch_size=128, shuffle=True
    )
    _ = utils.training_optimization(
        model= model, 
        epochs= 1,
        train_loader= heal_loader, 
        test_loader= retain_valid_dl,
        opt= "adam",
        device=device, 
        desc= "UNSIR repair step"
    )

    return model


# Selective Synaptic Dampening: https://github.com/if-loops/selective-synaptic-dampening
def ssd(
    args: argparse.Namespace,
    model: torch.nn.Module,
    unlearning_teacher: torch.nn.Module,
    unlearn_class: int,
    unlearn_loader: DataLoader,
    retain_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    num_channels: int,
    device: torch.device,
) -> torch.nn.Module:

    # Default parameters from repo
    parameters = {
        "lower_bound": 1,  # unused
        "exponent": 1,  # unused
        "magnitude_diff": None,  # unused
        "min_layer": -1,  # -1: all layers are available for modification
        "max_layer": -1,  # -1: all layers are available for modification
        "forget_threshold": 1,  # unused
        "dampening_constant": 1,  # Lambda from paper
        "selection_weighting": 10,  # Alpha from paper
    }

    # load the trained model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    pdr = ParameterPerturber(model, optimizer, device, parameters)

    model = model.eval()

    # Calculation of the forget set importances
    sample_importances = pdr.calc_importance(unlearn_loader)

    # Calculate the importances of D (see paper); this can also be done at any point before forgetting.
    original_importances = pdr.calc_importance(retain_loader)

    # Dampen selected parameters
    pdr.modify_weight(original_importances, sample_importances)

    return model