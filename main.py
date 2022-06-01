# coding: utf-8
import os
import argparse
import numpy as np

import matplotlib as mpl
from torch._C import device
mpl.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import Query2setFixedLenDataset
from model import Q2SUModel

def to_rgb(image):
    if(image.device!='cpu'):
        image = image.cpu()
    image = image.numpy()
    image = np.transpose(image, (0,2,3,1))
    if(image.shape[-1]==1):
        image = np.tile(image, (1,1,1,3))
    return image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train/test')
    parser.add_argument('--name', type=str, default='q2s', help='model name')
    parser.add_argument('--data_path', type=str, default='./dataset/fingerprint_query2set/train', help='data path of the train data')
    parser.add_argument('--value_path', type=str, default='./dataset/fingerprint_query2set/test', help='value path of the train data')
    parser.add_argument('--test_path', type=str, default='./dataset/fingerprint_query2set/test', help='test path of the train data')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint', help='chepoint path')
    parser.add_argument('--sample_path', type=str, default='./sample', help='sample path')
    parser.add_argument('--GPU', type=str, default='0', help='GPU ID for training')
    parser.add_argument('--epoch', type=int, default=5000, help='epoch')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--checkpoint_epoch', type=int, default=0, help='continue train')
    parser.add_argument('--disp_interval', type=int, default=10, help='interval for disp')
    parser.add_argument('--sample_interval', type=int, default=10, help='interval for sampling')
    parser.add_argument('--save_interval', type=int, default=100, help='interval for saving')
    args = parser.parse_args()

    if(not os.path.isdir(os.path.join(args.checkpoint_path, args.name))):
        os.makedirs(os.path.join(args.checkpoint_path, args.name))
    if(not os.path.isdir(os.path.join(args.sample_path, args.name))):
        os.makedirs(os.path.join(args.sample_path, args.name))
    
    if(args.mode=='train'):
        train(args)
    elif(args.mode=='test'):
        test(args)
    else:
        print('>>> ERROR! No such mode: ' + args.mode)

def train(args):
    # DEVICE
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    if torch.cuda.is_available():
        gpu_device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        gpu_device = torch.device("cpu")
    print('>>> [TORCH DEVICE]: ' + str(gpu_device))

    # Dataset
    train_dataset = Query2setFixedLenDataset(args.data_path)
    value_dataset = Query2setFixedLenDataset(args.value_path)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )
    value_sample_iterator = value_dataset.createIterator(128)

    # Model
    model = Q2SUModel()
    # Continue train
    checkpoint_path = os.path.join(args.checkpoint_path, args.name)
    if(args.checkpoint_epoch>0):
        model.load(checkpoint_path, args.checkpoint_epoch)
    # Move model to GPU
    model = model.to(gpu_device)

    # Epoch train
    for epoch in np.arange(args.checkpoint_epoch, args.epoch):
        total = len(train_dataset)
        if(total==0):
            print('>>> No training data was provided!')
        
        for batch_idx, (q, s, label) in enumerate(train_loader):
            model.train()

            # Move data to GPU
            q = q.to(gpu_device)
            s = [x.to(gpu_device) for x in s]
            label = label.to(gpu_device)

            # Forward
            pre_y, _, loss = model.process(q, s, label, 1.0)
            # Backward
            model.backward(loss)

            # Disp
            if((batch_idx+1)%args.disp_interval==0):
                print('Epoch:{}/{}, Ite:{}/{}, loss:{:.3f}'\
                    .format(epoch+1, args.epoch, batch_idx+1, total//args.batch_size, loss.item()))
        
        # Value
        with torch.no_grad():
            q, s, label = next(value_sample_iterator)
            q = q.to(gpu_device)
            s = [x.to(gpu_device) for x in s]
            label = label.to(gpu_device)
            pre_y, transformed_s, loss = model.process(q, s, label)
            pre_y = pre_y.detach()
            transformed_s = [x.detach() for x in transformed_s]
            loss = loss.detach()
            accuracy = torch.eq(pre_y.argmax(dim=1), label).float().mean()
            print('>>> Epoch:{}/{}, loss:{:.3f}, Accuracy:{:.3f}%'\
                .format(epoch+1, args.epoch, loss.item(), accuracy.item()*100))
        
        # Sample
        if((epoch+1)%args.sample_interval==0):
            sample_label = label[:8].cpu().numpy()
            sample_q = q[:8]
            sample_transformed_s = [x[:8] for x in transformed_s]
            gen_imgs = np.concatenate([to_rgb(sample_q)] + [to_rgb(x) for x in sample_transformed_s])
            s_len = len(sample_transformed_s)
            titles = ['Query'] + ['TransformedS_{}'.format(i+1) for i in range(s_len)]
            fig, axs = plt.subplots(s_len+1, 8, figsize=(8*3, (s_len+1)*3))
            cnt = 0
            for i in range(s_len+1):
                for j in range(8):
                    axs[i, j].imshow(gen_imgs[cnt])
                    axs[i, j].set_title(titles[i]+'-'+str(sample_label[j]))
                    axs[i, j].axis('off')
                    cnt += 1
            sample_path = os.path.join(args.sample_path, args.name)
            fig.savefig(os.path.join(sample_path, 'sample_{0:04d}.png'.format(epoch+1)), bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)

        # Save
        if((epoch+1)%args.save_interval==0):
            model.save(checkpoint_path, epoch+1)

def test(args):
    # DEVICE
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    if torch.cuda.is_available():
        gpu_device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        gpu_device = torch.device("cpu")
    print('>>> [TORCH DEVICE]: ' + str(gpu_device))

    # Dataset
    test_dataset = Query2setFixedLenDataset(args.test_path)
    total = len(test_dataset)
    if(total==0):
        print('>>> No testing data was provided!')
        return
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )

    # Load model
    model = Q2SUModel().to(gpu_device)
    model.load(os.path.join(args.checkpoint_path, args.name), args.epoch)

    # Test
    accuracys = []
    for batch_idx, (q, s, label) in enumerate(test_loader):
        model.eval()

        # Move data to GPU
        q = q.to(gpu_device)
        s = [x.to(gpu_device) for x in s]
        label = label.to(gpu_device)

        # Forward
        pre_y = model(q, s)
        # Accuracy
        accuracy = torch.eq(pre_y.argmax(dim=1), label).float().mean()
        accuracys.append(accuracy.item())
        # Disp
        print('Ite:{}/{}, Accuracy:{:.3f}%'\
            .format(batch_idx+1, total//args.batch_size, accuracy.item()*100))
    # Mean Accuracy
    accuracys = np.array(accuracys)
    print('>>> Test Mean Accuracy: {:.3f}%'.format(np.mean(accuracys)*100))

if __name__=='__main__':
    main()
