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
transformations = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
device = torch.device('cuda:0')
model = SixDRepNet(backbone_name='RepVGG-B1g2',
                       backbone_file='',
                       deploy=True,
                       pretrained=False)

detector = RetinaFace(gpu_id=0)
outDir=basFunc.MakeEmptyDir("outDir")
saved_state_dict = torch.load('models/6DRepNet_300W_LP_AFLW2000.pth', map_location='cpu')
#saved_state_dict = torch.load('output/best.pth', map_location='cpu')
if 'model_state_dict' in saved_state_dict:
    model.load_state_dict(saved_state_dict['model_state_dict'])
else:
    model.load_state_dict(saved_state_dict)
model.to(device)
model.eval()
def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
    """
    Prints the person's name and age.

    If the argument 'additional' is passed, then it is appended after the main info.

    Parameters
    ----------
    img : array
        Target image to be drawn on
    yaw : int
        yaw rotation
    pitch: int
        pitch rotation
    roll: int
        roll rotation
    tdx : int , optional
        shift on x axis
    tdy : int , optional
        shift on y axis
        
    Returns
    -------
    img : array
    """

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),4)

    return img
files=basFunc.getdatas("/data/valTestData","*.jpeg")
jsDir="/data/MMDetecTVT/jsonDir"
with torch.no_grad():
    for img_fp in files:
        print("Process the image: ", img_fp)
        img_ori = cv2.imread(img_fp)
        d,name,ftr=basFunc.GetfileDirNamefilter(img_fp)
        jsfile=os.path.join(jsDir,name+".json")
        if(not os.path.exists(jsfile)):continue
        namelst,box,w,h=ehl.ConvertOWNJson(jsfile)
        if(len(box)==0):continue
        rects=box
        for idx, rect in enumerate(rects):
            x_min = int(rect[0])
            y_min = int(rect[1])
            x_max = int(rect[2])
            y_max = int(rect[3])
            bbox_width = abs(x_max - x_min)
            bbox_height = abs(y_max - y_min)

            x_min = max(0, x_min-int(0.2*bbox_height))
            y_min = max(0, y_min-int(0.2*bbox_width))
            x_max = x_max+int(0.2*bbox_height)
            y_max = y_max+int(0.2*bbox_width)

            img = img_ori[y_min:y_max, x_min:x_max]
            img = Image.fromarray(img)
            img = img.convert('RGB')
            img = transformations(img)

            img = torch.Tensor(img[None, :]).to(device)

            R_pred = model(img)
            euler = utils.compute_euler_angles_from_rotation_matrices(
                R_pred)*180/np.pi
            p_pred_deg = euler[:, 0].cpu()
            y_pred_deg = euler[:, 1].cpu()
            r_pred_deg = euler[:, 2].cpu()
            draw_axis(img_ori, y_pred_deg, p_pred_deg, r_pred_deg, tdx=(x_min+x_max)/2, tdy=(y_min+y_max)/2, size=100)
        drawfile=os.path.join(outDir,name+ftr)
        cv2.imwrite(drawfile,img_ori)

# with torch.no_grad():
#     for file,d,name,ftr in basFunc.GetFileDirLst("img",filter="*.jpeg"):
#         frame=cv2.imread(file)
#         faces = detector(frame)
#         for box, landmarks, score in faces:

#             # Print the location of each face in this image
#             if score < .5:
#                 continue
#             x_min = int(box[0])
#             y_min = int(box[1])
#             x_max = int(box[2])
#             y_max = int(box[3])
#             bbox_width = abs(x_max - x_min)
#             bbox_height = abs(y_max - y_min)

#             x_min = max(0, x_min-int(0.2*bbox_height))
#             y_min = max(0, y_min-int(0.2*bbox_width))
#             x_max = x_max+int(0.2*bbox_height)
#             y_max = y_max+int(0.2*bbox_width)

#             img = frame[y_min:y_max, x_min:x_max]
#             img = Image.fromarray(img)
#             img = img.convert('RGB')
#             img = transformations(img)

#             img = torch.Tensor(img[None, :]).to(device)

#             R_pred = model(img)
#             euler = utils.compute_euler_angles_from_rotation_matrices(
#                 R_pred)*180/np.pi
#             p_pred_deg = euler[:, 0].cpu()
#             y_pred_deg = euler[:, 1].cpu()
#             r_pred_deg = euler[:, 2].cpu()
#             draw_axis(frame, y_pred_deg, p_pred_deg, r_pred_deg, tdx=(x_min+x_max)/2, tdy=(y_min+y_max)/2, size=100)
#         drawfile=os.path.join(outDir,name+ftr)
#         cv2.imwrite(drawfile,frame)