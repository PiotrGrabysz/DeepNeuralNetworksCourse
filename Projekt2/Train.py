import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from DataLoader import MyDataset
from MyTransforms import MyToPIL,MyRandomHorizontalFlip,MyRotate_90_270, MyToTensor
from MyTransforms import SelfAugmentation, MyNormalize

import numpy as np

SELF_AUGMENTATION = True

def train_step(inputs, masks, optimizer, criterion, unet, w, h):
    optimizer.zero_grad()
    outputs = unet(inputs)
    outputs = outputs.permute(0, 2, 3, 1)
    batch_size = outputs.shape[0]
    outputs = outputs.resize(batch_size*w*h)
    masks = masks.resize(batch_size*w*h)
    loss = criterion(outputs, masks)
    loss.backward()
    optimizer.step()
    return loss

def predict(unet,data, w, h, SELF_AUGMENTATION = True):
    images1 = data[0].to(device)
    images2 = data[1].to(device)
    images3 = data[2].to(device)
    images4 = data[3].to(device)
                       
    outputs1 = unet(images1).cpu()
    outputs2 = unet(images2).cpu()
    outputs3 = unet(images3).cpu()
    outputs4 = unet(images4).cpu()
    batch_size = outputs1.shape[0]
    final_outputs = torch.zeros(batch_size, 1, w, h)
    for k in range(batch_size):
        out1 = outputs1[k]
        out2 = TF.to_tensor(TF.hflip(TF.to_pil_image(outputs2[k])))
        out3 = TF.to_tensor(TF.rotate(TF.to_pil_image(outputs3[k]), angle = -90))
        out4 = TF.to_tensor(TF.rotate(TF.to_pil_image(outputs4[k]), angle = 90))
        final_outputs[k,:,:,:] = (out1+out2+out3+out4)/4
    return final_outputs.to(device), outputs1.to(device)

def get_test_loss(preds, masks, criterion, w, h):
    batch_size = preds.shape[0]
    
    preds = preds.permute(0, 2, 3, 1)
    preds = preds.resize(batch_size*w*h)
    masks = masks.resize(batch_size*w*h)
    loss = criterion(preds, masks)
    return loss.item()

def train(n_epochs, unet, optimizer, criterion, train_loader, test_loader,w, h,
          root, name):
    train_loss = []
    test_loss_aug = []
    test_loss = []
    test_acc = []
    max_acc = 0.
    IoU_metric = []
    for epoch in range(n_epochs):
        for data in train_loader:
            inputs, masks = data[0].to(device), data[1].to(device)
            loss = train_step(inputs, masks, optimizer, criterion, unet, w, h)
            train_loss.append(loss.item())
    
        correct = 0.
        total = 0.
        intersection = 0.
        union = 0.
        with torch.no_grad():
            for data in test_loader:
                pred_aug, pred = predict(unet,data, w, h)
                pred_aug = pred_aug.double()
                pred = pred.double()
                masks = data[4].to(device)
                test_loss.append(get_test_loss(pred, masks, criterion, w, h))
                test_loss_aug.append(get_test_loss(pred_aug, masks, criterion, w, h))
               
                final_output = (pred >= 0.5)
                batch_size = pred.shape[0]
                correct += torch.sum(final_output == masks)
                total += batch_size*w*h
                
                tmp = torch.sum(final_output * masks)
                intersection += tmp
                union +=torch.sum(final_output)+torch.sum(masks)-tmp
        test_acc.append(correct.cpu()/total)
        if(test_acc[-1] > max_acc):
            max_acc = test_acc[-1]
            path = root+'unet_weights_'+name+'.pth'
            torch.save(unet.state_dict(),path)
            
        IoU_metric.append(intersection.cpu()/union.cpu())
        
    path = root+'train_loss_'+name+'.npy'        
    np.save(path, train_loss)
    path = root+'test_loss_'+name+'.npy'        
    np.save(path, test_loss)
    path = root+'test_loss_aug_'+name+'.npy'        
    np.save(path, test_loss_aug)
    path = root+'test_acc_'+name+'.npy'        
    np.save(path, test_acc)
    path = root+'IoU_metric_'+name+'.npy'        
    np.save(path, IoU_metric)
    print("Net reached {}: accuracy on the test data".format(max_acc))    
            
train_dataset = MyDataset(img_file='/results/pgrabysz/projekt2/DATASET/gsn_img_uint8.npy',
                    mask_file='/results/pgrabysz/projekt2/DATASET/gsn_msk_uint8.npy',
                    transform = transforms.Compose([MyToPIL(),MyRandomHorizontalFlip(),
                                                    MyRotate_90_270(), MyToTensor()]))
    
if SELF_AUGMENTATION:
    test_dataset = MyDataset(img_file='/results/pgrabysz/projekt2/DATASET/test_gsn_image.npy',
                    mask_file='/results/pgrabysz/projekt2/DATASET/test_gsn_mask.npy',
                    transform = transforms.Compose([MyNormalize(), SelfAugmentation()]))
else:
    test_dataset = MyDataset(img_file='/results/pgrabysz/projekt2/DATASET/test_gsn_image.npy',
                    mask_file='/results/pgrabysz/projekt2/DATASET/test_gsn_mask.npy',
                    transform = MyNormalize())

train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=50,
        num_workers=0,
        shuffle=True)

test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=50,
        num_workers=0,
        shuffle=True)

from UNet import UNet
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
unet = UNet(in_channel = 3, out_channel = 1)
unet.to(device)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(unet.parameters(), lr = 0.001)
#optimizer = torch.optim.SGD(unet.parameters(), lr = 0.001, momentum = 0.99)

train(30, unet, optimizer, criterion,train_loader, test_loader,128,128,
      root = '/results/pgrabysz/', name = 'Adam_lr=0.001_60_epoch')



