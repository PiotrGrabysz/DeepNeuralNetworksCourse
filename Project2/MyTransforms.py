import numpy as np
import random
import torchvision.transforms.functional as TF

class MyToPIL(object):
    def __call__(self, sample):
        image, mask = sample[0], sample[1]
        return (TF.to_pil_image(image), TF.to_pil_image(mask))

class MyToTensor(object):
    def __call__(self, sample):
        image, mask = sample[0], sample[1]
        return (TF.to_tensor(image), TF.to_tensor(mask))
    
class MyNormalize(object):
    def __call__(self, sample):
        image, mask = sample[0], sample[1]
        return (image, mask/255.)

class MyRandomHorizontalFlip(object):
    def __call__(self, sample):
        image, mask = sample[0], sample[1]        
        if random.randint(0,1) == 1:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        return (image, mask)
    
class MyRotate_90_270(object):
    def __call__(self, sample):
        image, mask = sample[0], sample[1]        
        rotate = random.randint(0,2)
        if rotate == 1:
            image = TF.rotate(image, 90)
            mask = TF.rotate(mask, 90)
        if rotate == 2:
            image = TF.rotate(image, -90)
            mask = TF.rotate(mask, -90)
        return (image, mask)
    
class SelfAugmentation(object):
    def __call__(self, sample):
        image = TF.to_pil_image(sample[0])  
        image2 = TF.hflip(image)
        image3 = TF.rotate(image, angle = 90)
        image4 = TF.rotate(image, angle = -90)
        
        masks = sample[1]
        
        return (TF.to_tensor(image), TF.to_tensor(image2),
                TF.to_tensor(image3),TF.to_tensor(image4), TF.to_tensor(masks))

class InverseTransform(object):
    def __call__(self, masks):
        mask1 = masks[0]  
        mask2 = TF.hflip(TF.to_pil_image(masks[1]))
        mask3 = TF.rotate(TF.to_pil_image(masks[2]), angle = -90)
        mask4 = TF.rotate(TF.to_pil_image(masks[3]), angle = 90)
               
        return (mask1, TF.to_tensor(mask2),
                TF.to_tensor(mask3), TF.to_tensor(mask4))
    

