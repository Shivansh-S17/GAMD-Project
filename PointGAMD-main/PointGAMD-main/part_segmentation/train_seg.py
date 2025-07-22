
from colorama import Fore
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from torch.autograd import Variable
from scripts.h5_dataset import Dataset
from models.Descriptor_segmentation_nodescriptors import Model
from scripts.functions import save_checkpoint
from tqdm import tqdm
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser()

    # naming / file handling
    parser.add_argument(
        '--run_name', type=str, default='Seg_1', help='training run name')
    parser.add_argument(
        '--desc', type=str, default='Trail 1.', help='description')
    parser.add_argument('--data_root', type=str, default='',
                        help='data root containing the data set folder (point clouds)')
    parser.add_argument('--logdir', type=str,
                        default='runs', help='training log folder')
    parser.add_argument('--refine', type=bool,
                        default=False, help='train again a pretrained model or not')
    parser.add_argument('--loaddir', type=str,
                        default='', help='path to model file (.pth.tar) to finetune')

    # dataset
    parser.add_argument('--dataset', type=str,
                        default='shapenetpart', help='shapenetpart')
    parser.add_argument('--class_choice', type=str,
                        default=None, help='airplane, vase .. etc')
    parser.add_argument('--augment', type=bool,
                        default=True, help='Apply jitter and rotation to data for training')
    parser.add_argument('--shuffle', type=bool, default= True,
                        help='Shuffle Dataset or not')

    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points of point clouds')
    parser.add_argument('--device', type=str, default='cpu',
                        help='"gpu" or "cpu"')
    parser.add_argument('--pool_factor', type=int, default=2,
                        help='pool factor for pooling graph (either 2 or 4)')
    parser.add_argument('--k_neighbours', type=int, default=24,
                        help='number of neighbours to consider for a point')

    # training parameters
    parser.add_argument('--nepochs', type=int, default=100,
                        help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='input batch size')
    parser.add_argument('--workers', type=int, default=1,
                        help='number of data loading workers - 0 means same thread as main execution')
    parser.add_argument('--cache_capacity', type=int, default=600,
                        help='Max. number of dataset elements (usually shapes) to hold in the cache at the same time.')
    parser.add_argument('--seed', type=int,
                        default=3627473, help='manual seed')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='gradient descent momentum')

    # model hyperparameters
    parser.add_argument('--trans_loss', type=bool,
                        default=False, help='use transform loss')
    parser.add_argument('--alpha', type=int,
                        default=0.001, help='coeficient for transform loss')

    return parser.parse_args()


def check_path_existance(log_dirname):
    if os.path.exists(log_dirname):
        print(Fore.GREEN + "log directory : ", log_dirname)
        return
    else:
        os.mkdir(log_dirname)
        os.mkdir(log_dirname+'/val')
        os.mkdir(log_dirname+'/test')
        print(Fore.GREEN + "Created log directory :  ", log_dirname)

def shapenet_class_IoU(confusion):
    x = confusion.shape[0]
    true_positive = np.zeros(x)
    false_positive = np.zeros(x)
    false_negative = np.zeros(x)
    union = np.zeros(x)

    for i in range(x):
        true_positive[i] = confusion[i, i]
        false_positive[i] = np.sum(confusion[:, i]) - true_positive[i]
        false_negative[i] = np.sum(confusion[i, :]) - true_positive[i]
        union[i] = true_positive[i] + false_positive[i] + false_negative[i]

    object_acc = np.zeros(16)
    parts = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
    k = 0
    for i in range(16):
        j = parts[i]
        object_acc[i] = (true_positive[k:k+j].sum()) / (union[k:k+j].sum())
        k += j

    class_miou = round((object_acc.sum()/16)*100, 2)
    print('Class Mean Accuracy : ', class_miou)

    instance_miou = round((true_positive.sum()/union.sum())*100, 2)
    print('Instance Mean Accuracy : ', instance_miou)

    dict = {'class_iou': object_acc,
            'class_miou': class_miou,
            'instance_miou': instance_miou}

    return dict

def get_data(opt, train=True):
    # create train and test dataset loaders

    if train:
        if(opt.dataset == 'shapenetpart'):
            dataset = Dataset(root=opt.data_root,
                                dataset_name=opt.dataset,
                                class_choice= opt.class_choice,
                                load_name = False,
                                load_file = False,
                                segmentation=True,
                                random_rotate  = False,
                            random_jitter = False,
                            random_translate = False,
                            num_points=opt.num_points)
            print(Fore.GREEN+"ShapeNetpart dataset")
            return dataset
        else:
            print(Fore.RED+"Wrong dataset choosen! : ",opt.data_root)
            raise AssertionError
    else:
        if(opt.dataset == 'shapenetpart'):
            dataset = Dataset(root=opt.data_root,
                                dataset_name=opt.dataset,
                                class_choice= opt.class_choice,
                                load_name = False,
                                load_file = False,
                                segmentation=True,
                                random_rotate  = False,
                            random_jitter = False,
                            random_translate = False,
                            num_points=opt.num_points,
                            split='test')
            
            print(Fore.GREEN+"ShapeNetpart Test dataset")
            return dataset
        else:
            print(Fore.RED+"Wrong dataset choosen! : ",opt.data_root)
            raise AssertionError 
            

def train_adpnet(opt):

    log_dirname = os.path.join(opt.logdir, opt.run_name)
    params_filename = os.path.join(log_dirname, "config.txt")

    check_path_existance(log_dirname)

    if opt.seed < 0:
        opt.seed = random.randint(1, 10000)
    print(Fore.GREEN+"Random Seed: %d" % (opt.seed))
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    # create train and test dataset loaders

    train_dataset = get_data(opt, train=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        num_workers=int(opt.workers),
        shuffle=opt.shuffle)
    
    if opt.class_choice == None:
        num_classes = 50
    else:
        num_classes = np.unique(train_dataset[0][2]).shape[0]
    
    adpnet = Model(classes=num_classes,
                  k_neighbours=opt.k_neighbours,
                  index = 0
                    )

    if opt.refine:
        if os.path.exists(opt.loaddir):
            last_model = torch.load(opt.loaddir)
            adpnet.load_state_dict(last_model['model_state_dict'])
            max_acc = last_model['accuracy']
            print(Fore.GREEN+"Model Loaded sucessfully : ", opt.loaddir)
        else:
            print(Fore.RED+"invalid load path for model params")
            return

    test_dataset = get_data(opt, train=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        num_workers=int(opt.workers),
        shuffle=False,
        drop_last = False)

    print(Fore.GREEN+'training set: %d objects (in %d batches) /n' %
          (len(train_dataset), len(train_dataloader)))
    print(Fore.GREEN+'testing set: %d objects (in %d batches) /n' %
          (len(test_dataset), len(test_dataloader)))

    optimizer = torch.optim.SGD(adpnet.parameters(), lr=opt.lr, momentum=opt.momentum)
    # milestones in number of optimizer iterations
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1, verbose = True)

    adpnet = adpnet.cuda()

    # save parameters
    if (os.path.exists(params_filename)):
        torch.save(opt, params_filename)
    else:
        with open(params_filename, 'w') as fp:
            fp.close()
        torch.save(opt, params_filename)

    criterion = torch.nn.CrossEntropyLoss()

    train_loss = []
    train_acc = []
    val_acc = []
    test_acc = []

    if opt.refine:
        print(max_acc)
    else:
        max_acc = 0
    
    print(Fore.GREEN)
    for epoch in range(1, opt.nepochs + 1):
        print(Fore.GREEN)
        num_train_examples = train_dataset.__len__()
        num_train_batch = len(train_dataloader)
        num_test_examples = test_dataset.__len__()
        num_test_batch = len(test_dataloader)

        epoch_loss, correct, eval_correct, test_correct = 0, 0, 0, 0
        test_confusion = torch.zeros((num_classes, num_classes),dtype=torch.int)

        progress_bar = tqdm(
            range(num_train_batch),
            desc=f"Training Epoch {epoch}/{opt.nepochs}"
        )

        for batch_idx, (data, _, labels) in enumerate(train_dataloader):
            # data,_,labels = next(iter(train_dataloader))
            data = data.transpose(1,2).cuda()
            labels = labels.squeeze(1).cuda()

            adpnet.train()
            optimizer.zero_grad()

            prediction = adpnet(data)
            loss = criterion(prediction, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            y_pred_softmax = torch.log_softmax(prediction, dim = 1)
            preds = y_pred_softmax.data.max(1)[1]
            # OVerall acc
            correct += preds.eq(labels.data).sum().item()

        scheduler.step()

        epoch_loss = round(epoch_loss / num_train_batch, 5)
        epoch_accuracy = round((correct / (num_train_examples*opt.num_points))*100, 2)

        print(Fore.CYAN+"Loss : ", epoch_loss)
        print(Fore.CYAN+"accuracy : ", epoch_accuracy)

                   
        train_loss.append(epoch_loss)
        train_acc.append(epoch_accuracy)
        if epoch_accuracy > max_acc:
            max_acc = epoch_accuracy
            state = {'epoch': epoch,
                    'model_state_dict': adpnet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': epoch_accuracy,
                    'loss': epoch_loss,
                    }
            save_checkpoint(state, log_dirname, opt.run_name, '.pth.tar')
                        
        # Testing 
        if(epoch > 0 ):
            print(Fore.WHITE)
            adpnet.eval()
            with torch.no_grad():
                for batch_idx, (data, _, labels) in enumerate(test_dataloader):
                    data = data.transpose(1,2).cuda()
                    labels = labels.squeeze(1).cuda()
                    prediction = adpnet(data)

                    y_pred_softmax = torch.log_softmax(prediction, dim = 1)
                    preds = y_pred_softmax.data.max(1)[1]
                    # OVerall iacc
                    test_correct += preds.eq(labels.data).sum().item()
                    # Class accs
                    for t, p in zip(labels.view(-1), preds.view(-1)):
                        test_confusion[t.long(), p.long()] += 1

            x = shapenet_class_IoU(test_confusion.detach().cpu().numpy())

            test_acc_s = round((test_correct / (num_test_examples*opt.num_points))*100,3)
            print("==> Test accuracy : ", test_acc_s)
            test_acc.append(test_acc_s)

            test_state = {'epoch': epoch,
                'model_state_dict': adpnet.state_dict(),
                'test_accuracy': test_acc_s,
                'accuracy': epoch_accuracy,
                'loss': epoch_loss,
                'confusion' : test_confusion,
                'iou':x
                }
            with open(log_dirname+'/test/'+'model_pt_'+str(x['class_miou'])+'_'+str(x['instance_miou'])+'.pth.tar', 'w') as fp:
                fp.close()
            torch.save(test_state, log_dirname+'/test/'+'model_pt_'+str(x['class_miou'])+'_'+str(x['instance_miou'])+'.pth.tar')
        

    if (os.path.exists == log_dirname+'/'+'train_acc.npy'):
            a = np.load(log_dirname+'/'+'train_acc.npy')
            a = a.append(train_acc)
            np.save(log_dirname+'/'+'train_acc.npy', a)
            print(Fore.GREEN+">>> Saved train_acc.npy")
    else:
        np.save(log_dirname+'/'+'train_acc.npy', train_acc)
        print(Fore.GREEN+">>> Saved train_acc.npy")

    if (os.path.exists == log_dirname+'/'+'train_loss.npy'):
        a = np.load(log_dirname+'/'+'train_loss.npy')
        a = a.append(train_loss)
        np.save(log_dirname+'/'+'train_loss.npy', a)
        print(Fore.GREEN+">>> Saved train_loss.npy")
    else:
        np.save(log_dirname+'/'+'train_loss.npy', train_loss)
        print(Fore.GREEN+">>> Saved train_loss.npy")

    if (os.path.exists == log_dirname+'/'+'test_acc.npy'):
        a = np.load(log_dirname+'/'+'test_acc.npy')
        a = a.append(test_acc)
        np.save(log_dirname+'/'+'test_acc.npy', a)
        print(Fore.GREEN+">>> Saved test_acc.npy")
    else:
        np.save(log_dirname+'/'+'test_acc.npy', test_acc)
        print(Fore.GREEN+">>> Saved test_acc.npy")

        

if __name__ == '__main__':
    train_opt = parse_arguments()
    train_adpnet(train_opt)