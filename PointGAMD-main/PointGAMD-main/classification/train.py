import os
import torch
import random
import numpy as np
from tqdm import tqdm
from colorama import Fore
import time
from model import Model
from scripts.h5_dataset import Dataset
from scripts.functions import save_checkpoint

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def check_path_existance(log_dirname):
    if not os.path.exists(log_dirname):
        os.makedirs(os.path.join(log_dirname, 'val'), exist_ok=True)
        os.makedirs(os.path.join(log_dirname, 'test'), exist_ok=True)
        print(Fore.GREEN + "Created log directory:", log_dirname)
    else:
        print(Fore.GREEN + "Log directory exists:", log_dirname)

def compute_accuracy(conf_matrix):
    total = np.sum(conf_matrix)
    overall_acc = np.trace(conf_matrix) / total
    class_acc = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    mean_acc = np.mean(class_acc)
    return round(overall_acc * 100, 2), round(mean_acc * 100, 2)

def get_data(opt, train=True):
    dataset = Dataset(
        root=opt.data_root,
        split='train' if train else 'test',
        random_jitter=opt.augment,
        random_rotate=opt.augment
    )
    print(Fore.GREEN + f"{'Train' if train else 'Test'} dataset loaded:", len(dataset))
    return dataset


def train_adpnet(opt):
    log_dir = os.path.join(opt.logdir, opt.run_name)
    check_path_existance(log_dir)

    # Dataset loading
    train_set = get_data(opt, train=True)
    test_set = get_data(opt, train=False)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers
    )

    # Model
    model = Model(classes=40, k_neighbours=opt.k_n, index=opt.cuda_idx).cuda(opt.cuda_idx)
    print(Fore.GREEN + f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Optimizer & scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepochs, eta_min=opt.lr/100)

    criterion = torch.nn.CrossEntropyLoss()
    max_acc = 0
    max_test = 0

    for epoch in range(1, opt.nepochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        for data, label in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            data = data.transpose(1, 2).cuda(opt.cuda_idx)  # [B, 3, N]
            label = label.squeeze(1).cuda(opt.cuda_idx)

            optimizer.zero_grad()
            pred, _ = model(data)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += pred.max(1)[1].eq(label).sum().item()

        scheduler.step()
        acc = round((correct / len(train_set)) * 100, 2)
        print(Fore.CYAN + f"Train Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")

        # --- Testing ---
        model.eval()
        test_correct = 0
        conf_matrix = np.zeros((40, 40))
        with torch.no_grad():
            for data, label in tqdm(test_loader, desc=f"Epoch {epoch} [Test]"):
                data = data.transpose(1, 2).cuda(opt.cuda_idx)
                label = label.squeeze(1).cuda(opt.cuda_idx)
                pred, _ = model(data)
                pred_label = pred.max(1)[1]
                test_correct += pred_label.eq(label).sum().item()

                for t, p in zip(label.view(-1), pred_label.view(-1)):
                    conf_matrix[t.item(), p.item()] += 1

        test_acc, mean_acc = compute_accuracy(conf_matrix)
        print(Fore.YELLOW + f"Test Acc: {test_acc:.2f}%, Mean Acc: {mean_acc:.2f}%")

        if test_acc > max_test:
            max_test = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'test_accuracy': test_acc,
                'accuracy': acc
            }, os.path.join(log_dir, 'test', f'model_epoch_{epoch}_acc_{test_acc:.2f}.pth.tar'))

        if acc > max_acc:
            max_acc = acc
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': acc,
                'test_accuracy': test_acc
            }, log_dir, opt.run_name, '.pth.tar')

    print(Fore.GREEN + f"Max Train Accuracy: {max_acc:.2f}%")
    print(Fore.GREEN + f"Max Test Accuracy: {max_test:.2f}%")

# ---- Run from inside VS Code / manually ----
import time  # Add at the top of the file if not already

if __name__ == '__main__':
    class Args:
        dataset = 'modelnet40'
        data_root = 'dataset/h5'
        task = 'class'
        device = 'gpu'
        cuda_idx = 0
        run_name = 'fps_adapt'
        augment = True
        batch_size = 16
        nepochs = 50
        k_n = 20
        refine = False
        loaddir = ''
        logdir = 'runs'
        shuffle = True
        num_points = 2048
        final_dim = 1024
        pool_factor = 2
        workers = 0
        cache_capacity = 600
        seed = -1
        lr = 0.001
        momentum = 0.9
        trans_loss = False
        alpha = 0.001
        desc = 'FPS only run'
        saveinterval = 5

    # ⏱ Start timer
    start_time = time.time()

    train_adpnet(Args())

    # ⏱ End timer
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n✅ Total training time: {total_time / 60:.2f} minutes ({total_time:.2f} seconds)")
