import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from DataLoader import MyDataset
from MyTransforms import SelfAugmentation, MyNormalize

import numpy as np

SELF_AUGMENTATION = True
TRAINING_SET = True

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
    
if SELF_AUGMENTATION:
    train_dataset = MyDataset(img_file='../results/pgrabysz/projekt2/DATASET/gsn_img_uint8.npy',
                    mask_file='../results/pgrabysz/projekt2/DATASET/gsn_msk_uint8.npy',
                    transform = transforms.Compose([MyNormalize(), SelfAugmentation()]))
    
    test_dataset = MyDataset(img_file='../results/pgrabysz/projekt2/DATASET/test_gsn_image.npy',
                    mask_file='../results/pgrabysz/projekt2/DATASET/test_gsn_mask.npy',
                    transform = transforms.Compose([MyNormalize(), SelfAugmentation()]))
else:
    train_dataset = MyDataset(img_file='../results/pgrabysz/projekt2/DATASET/gsn_img_uint8.npy',
                    mask_file='../results/pgrabysz/projekt2/DATASET/gsn_msk_uint8.npy',
                    transform = MyNormalize())
    test_dataset = MyDataset(img_file='/results/pgrabysz/projekt2/DATASET/test_gsn_image.npy',
                    mask_file='../results/pgrabysz/projekt2/DATASET/test_gsn_mask.npy',
                    transform = MyNormalize())

if TRAINING_SET:
    data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=50,
            num_workers=0,
            shuffle=False)
else:
    data_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=50,
            num_workers=0,
            shuffle=False)

from UNet import UNet
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

unet = UNet(in_channel = 3, out_channel = 1)
unet.to(device)
unet_name = '../results/pgrabysz/unet_weights_Adam_lr=0.001_next_30_epoch.pth'
unet.load_state_dict(torch.load(unet_name,map_location=torch.device('cpu')))
criterion = torch.nn.BCELoss()

images = []
masks = []
preds = []
losses = []
batch_size = 50
w = h = 128

with torch.no_grad():
    for i, data in enumerate(data_loader):
        print(i)
        images1 = data[0].to(device)
        images2 = data[1].to(device)
        images3 = data[2].to(device)
        images4 = data[3].to(device)
        
        outputs1 = unet(images1).cpu()
        outputs2 = unet(images2).cpu()
        outputs3 = unet(images3).cpu()
        outputs4 = unet(images4).cpu()
        for k in range(batch_size):
            out1 = outputs1[k]
            out2 = TF.to_tensor(TF.hflip(TF.to_pil_image(outputs2[k])))
            out3 = TF.to_tensor(TF.rotate(TF.to_pil_image(outputs3[k]), angle = -90))
            out4 = TF.to_tensor(TF.rotate(TF.to_pil_image(outputs4[k]), angle = 90))
            
            pred = (out1+out2+out3+out4)/4
            pred = pred.permute(1,2,0)
            mask = (data[4])[k]
            mask = mask.permute(1,2,0)
            
            images.append(images1[k].permute(1,2,0).cpu().numpy())
            preds.append(pred.numpy())
            masks.append(mask.numpy())
            
            pred = (pred.resize(w*h)).double()
            mask = mask.resize(w*h)
            loss = criterion(pred, mask)
            losses.append(loss.item())

if TRAINING_SET:    
    np.save('../results/pgrabysz/images_train.npy', images)
    np.save('../results/pgrabysz/preds_train.npy', preds)
    np.save('../results/pgrabysz/masks_train.npy', masks)
    np.save('../results/pgrabysz/losses_train.npy', losses)
    
else:    
    np.save('/results/pgrabysz/images.npy', images)
    np.save('/results/pgrabysz/preds.npy', preds)
    np.save('/results/pgrabysz/masks.npy', masks)
    np.save('/results/pgrabysz/losses.npy', losses)