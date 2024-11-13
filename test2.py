import os,sys
import cvSDK.BasicUseFunc as basFunc
import cvSDK.EhualuInterFace as ehl
from face_detection import RetinaFace
from torchvision import transforms
from sixdrepnet.model import SixDRepNet
import torch,cv2
from PIL import Image
import sixdrepnet.utils  as utils
import numpy as np
from math import *
from sixdrepnet.train import Validate
from sixdrepnet.transform import *
import sixdrepnet.datasets as datasets
device = torch.device('cuda:0')
model = SixDRepNet(backbone_name='RepVGG-B1g2',
                       backbone_file='',
                       deploy=True,
                       pretrained=False)
saved_state_dict = torch.load('models/6DRepNet_300W_LP_AFLW2000.pth', map_location='cpu')
if 'model_state_dict' in saved_state_dict:
    model.load_state_dict(saved_state_dict['model_state_dict'])
else:
    model.load_state_dict(saved_state_dict)
model.to(device)
model.eval()
val_transform=transforms.Compose([
        letterBox(224),
        InputToTensor(),
        Normlize()
    ])
val_dataset=datasets.getDataset("chenzhuo","/data/headDir/val","",val_transform)
vtrain_loader=torch.utils.data.DataLoader(dataset=val_dataset,batch_size=100,shuffle=False,num_workers=8)
Validate(model,vtrain_loader)