import enum
import BasicUseFunc as basFunc
import numpy as np
import os,sys 
import BasicDrawing as basDraw
from torchvision.ops.boxes import nms as nms_torch
import DataIO as dio
from enum import Enum
import matplotlib.image as imgUse
import PIL.Image as pilimg
import EhualuInterFace as ehl
import cv2
def GetColor(i):
    color=[[0,0,0],
           [255,0,0],
           [0,255,0],
           [0,0,255],
           [128,0,0],
           [0,128,0],
           [0,0,128],
           [64,0,0],
           [0,64,0],
           [0,0,64],
           [255,255,0],
           [0,255,255],
           [255,0,255],
           [128,0,128],
           [0,128,128],
           [128,0,128],
           [64,64,0],
           [0,64,64],
           [64,0,64],
           [255,0,128],
           [0,255,128],
           [128,0,255],
           [128,0,255],
           [0,128,255],
           [255,0,128]]
    return color[i%len(color)]
def BlendOriImage(file,orifile,maskfile,height,width):
    mask=dio.GetExpandImageData(maskfile,height,width)
    mask=np.array(mask,np.uint8)
    mask=mask[:,:,0:1]
    mask=np.reshape(mask,[mask.shape[0],mask.shape[1]])
    orimg=dio.GetExpandImageMatData(orifile,height,width)
    orimg=np.asarray(orimg,np.float32)
    image=orimg.copy()
    for i in range(255):
        smask=mask==i
        check=np.linalg.sum(smask)
        if(check==0):break
        clr=GetColor(i)
        image[:,:,0][smask]=clr[0]
        image[:,:,1][smask]=clr[1]
        image[:,:,2][smask]=clr[2]
    image=(image+orimg)/2
    image=np.asarray(image,np.uint8)
    imgUse.imsave(file,image)
    return image
def GetPolyBox(ptlsts):
    boxes=[]
    for i,ptlst in enumerate(ptlsts):
        pts=np.array(ptlst,np.float32)
        xmin,xmax=pts[...,0].min(),pts[...,0].max()
        ymin,ymax=pts[...,1].min(),pts[...,1].max()
        boxes.append([xmin,ymin,xmax,ymax])
    return boxes
def GetMaskBox(masks,classlst=None):
    boxes=[]
    namelst=[]
    for i in range(masks.shape[-1]):
        m = masks[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
            boxes.append([x1,y1, x2,y2])
            if classlst is not None:
                namelst.append(classlst[i])
    if len(namelst)==0:
        namelst=None
    return boxes,namelst
def GetMaskAreaBox(masks,classlst=None):
    boxes=[]
    namelst=[]
    for i in range(masks.shape[-1]):
        m = masks[:, :, i]
        bmp=np.zeros(m.shape,np.uint8)
        bmp[m>0]=255
        num,label=cv2.connectedComponents(bmp)
        for j in range(num):
            if j==0:continue
            msk=label==j
            wmax=msk.sum(1).max()
            hmax=msk.sum(0).max()
            if wmax<4 or hmax<4:continue
            clslst=None
            if classlst is not None:
                clslst=[classlst[i]]
            boxs,names=GetMaskBox(msk[...,None],clslst) 
            if classlst is not None:
                namelst.extend(names)
            boxes.extend(boxs)
    if len(namelst)==0:
        namelst=None
    return boxes,namelst    
#如果出现重复的区域,后面的mask会覆盖前面的mask
def ConvertPolylstToOneMask(ptlsts,w,h):
    masks=[]    
    for i,ptlst in enumerate(ptlsts):
        mask=np.zeros([h,w,3],np.uint8)
        pts=np.array(ptlst,np.int32)
        mask=cv2.fillPoly(mask,[pts],(1,1,1))
        masks.append(mask[...,0][...,None])
    masks=np.concatenate(masks,-1)
    return masks
#专门配合ehl.ConvertOwnSegJson
def ConvertPolylstToOneMaskFromDetSeg(nptslst,namelst,clsOrder,w,h):
    cntlst=[clsOrder.index(n)+1 for n in namelst]
    img=np.zeros([h,w],np.uint8)
    for i,npt in enumerate(nptslst):
        mask=ConvertPolylstToOneMask(npt,w,h).sum(-1)>0
        img[mask]=cntlst[i]
    return img
def ConvertOrderPolylstToOneMask(ptlsts,w,h,clslst,clsOrder):
    masks=ConvertPolylstToOneMask(ptlsts,w,h)
    clsData=[[] for i in range(len(clsOrder))]
    for i,cls in enumerate(clslst):
        idx=clsOrder.index(cls)
        clsData[idx].append(masks[...,i])
    finalmsk=[]
    for i in range(len(clsOrder)):
        if len(clsData[i])==0:
            finalmsk.append(clsData[i])
        else:
            s=np.stack(clsData[i])
            s=s.sum(0)
            msk=s>0
            finalmsk.append(msk)
    return finalmsk
def BlendMasksToImage(img,masks,mskNum=1):
    image=np.array(img,np.float32)
    for j in range(masks.shape[-1]):
        smaks=masks[...,j]==mskNum
        clr=GetColor(j)
        image[smaks]=(np.array(clr,np.float32)+image[smaks])/2
        image=np.where(image>255.0,255.0,image)
    image=image.astype(np.uint8)
    return image
def BlendMasksLogitsToImage(img,masks,logits,minratio=0.1,mskratio=0.8,mskNum=1):
    image=np.array(img,np.float32)
    for j in range(masks.shape[-1]):
        smaks=masks[...,j]==mskNum
        clr=GetColor(j)
        l=logits[...,j].copy()
        l*=(1-minratio)*mskratio
        l+=minratio*mskratio
        r=l[...,None]*np.array(clr,np.float32)[None,None,:]
        image[smaks]=r[smaks]+image[smaks]*(1.0-mskratio)
        image=np.where(image>255.0,255.0,image)
    image=image.astype(np.uint8)
    return image
def BlendMasksToImageFile(file,outfile,masks,mskNum=1):
    image=dio.GetOriginImageData(file)
    image=BlendMasksToImage(image,masks,mskNum)
    imgUse.imsave(outfile,image)
def ConvertINToMask(pngdata,plus=0):
    maxNum=pngdata.max()+1
    masks=[]
    for i in range(maxNum):
        mask=np.where(pngdata==i+plus,1,0)
        masks.append(mask[...,None])
    masks=np.concatenate(masks,-1)
    return masks
def ConvertINToMaskClsNum(pngdata,clsNum,plus=0):
    maxNum=clsNum
    masks=[]
    for i in range(maxNum):
        mask=np.where(pngdata==i+plus,1,0)
        masks.append(mask[...,None])
    masks=np.concatenate(masks,-1)
    return masks
def ShowMasksBoxToImage(img,masks,classlst=None):
    img=BlendMasksToImage(img,masks)
    boxes,namelst=GetMaskAreaBox(masks,classlst)
    im=pilimg.fromarray(img)
    im=basDraw.DrawImageRectangles(im,boxes,namelst=namelst)
    img=np.array(im,np.uint8)
    return img
def ShowMasksLogitsBoxToImage(img,masks,logits,classlst=None):
    img=BlendMasksLogitsToImage(img,masks,logits)
    boxes,namelst=GetMaskAreaBox(masks,classlst)
    im=pilimg.fromarray(img)
    im=basDraw.DrawImageRectangles(im,boxes,namelst=namelst)
    img=np.array(im,np.uint8)
    return img
def ShowMasksBoxToImageFile(outfile,img,masks,classlst=None):
    img=ShowMasksBoxToImage(img,masks,classlst)
    imgUse.imsave(outfile,img)
def ShowMasksLogitsBoxToImageFile(outfile,img,masks,logits,classlst=None):
    img=ShowMasksLogitsBoxToImage(img,masks,logits,classlst)
    imgUse.imsave(outfile,img)
def ConvertOrderMaskToInt(ordmasks,w,h):
    res=np.zeros([h,w],np.uint8)
    for i in range(len(ordmasks)):
        if type(ordmasks[i]) is list:continue
        res[ordmasks[i]]=i+1
    return res
def ShowMaskIntBoxToImage(img,maskint,classlst):
    masks=ConvertINToMaskClsNum(maskint,len(classlst))
    img=ShowMasksBoxToImage(img,masks,classlst)
    return img
def ShowMaskIntBoxToImageFile(outfile,img,maskint,classlst):
    img=ShowMaskIntBoxToImage(img,maskint,classlst)
    imgUse.imsave(outfile,img)
def ShowMaskIntBoxToImageCopyFile(outfile,imgfile,maskint,classlst):
    img=dio.GetOriginImageData(imgfile)
    ShowMaskIntBoxToImageFile(outfile,img,maskint,classlst)
def ShowClassPalettle(filename,cls,plt,colorsize=60):
    assert(len(cls)==len(plt))
    num=len(cls)
    n=int(math.sqrt(num))+1
    m=num//n+1
    img=np.zeros([m*colorsize,n*colorsize,3],np.uint8)
    for i in range(len(cls)):
        target=(i//n,i%n)
        pos=[target[0]*colorsize,target[1]*colorsize]
        img[pos[1]:pos[1]+colorsize,pos[0]:pos[0]+colorsize]=np.array(plt[i])[None,None,...]
    img=pilimg.fromarray(img)
    wposlst=[]
    wtxtlst=[]
    bposlst=[]
    btxtlst=[]
    for i in range(len(cls)):
        target=(i//n,i%n)
        pos=[target[0]*colorsize+5,target[1]*colorsize+5]
        clr=max(plt[i])
        if clr<128:
            wposlst.append(pos)
            wtxtlst.append(cls[i])
        else:
            bposlst.append(pos)
            btxtlst.append(cls[i])        
    img=basDraw.DrawImageText(img,wposlst,wtxtlst,color='#ffffff')
    img=basDraw.DrawImageText(img,bposlst,btxtlst,color='#000000')
    img=np.array(img,np.uint8)    
    imgUse.imsave(filename,img)
def DealContours(contours,buffer):
    if len(contours)<=1:return contours,[]
    zs=np.zeros(buffer.shape[:2])
    areas = [cv2.contourArea(contours[i]) for i in range(len(contours))]
    sortlst=sorted(zip(areas,contours),key=lambda x:x[0],reverse=True)
    areas,contours=zip(*sortlst)
    nconts,oconts=[],[]
    for i in range(len(contours)):
        cont=contours[i]
        tempzs=np.zeros_like(zs)
        cont=cont.squeeze(1)
        cv2.fillPoly(tempzs,[cont],255)
        tempzs=np.where(tempzs>1,1,0)
        zs+=tempzs
        msk=zs>1
        tmsk=tempzs>0
        msk=np.logical_and(msk,tmsk)
        if msk.sum()==tmsk.sum():
            oconts.append(contours[i])
        else:
            nconts.append(contours[i])
    return nconts,oconts
def Region2Poly(pred,epsilon,classes,areathr=1000):
    nptlst,ndxlst,groupids=[],[],[]
    clsNum=len(classes)
    for j in range(1,clsNum):
        buffer=np.zeros(list(pred.shape[:2]),np.uint8)
        msk=pred==j
        buffer[msk]=255
        contours,hierarchy=cv2.findContours(buffer,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours,backcontour=DealContours(contours,buffer)
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area<areathr:
                continue
            polysts = cv2.approxPolyDP(contours[i],epsilon,True)
            polysts=polysts.squeeze(1).tolist()
            nptlst.append([polysts])
            ndxlst.append(classes[j])
            groupids.append(None)
        for i in range(len(backcontour)):
            area = cv2.contourArea(backcontour[i])
            if area<areathr:
                continue
            polysts = cv2.approxPolyDP(backcontour[i],epsilon,True)
            polysts=polysts.squeeze(1).tolist()
            nptlst.append([polysts])
            ndxlst.append(classes[0])
            groupids.append(j)
    return nptlst,ndxlst,groupids
def WriteOwnSegJson(nptlst,ndxlst,groupids,jsfile,imgfile=None,w=-1,h=-1):
    if w<0 or h<0:
        img=dio.GetOriginImageData(imgfile)
        h=img.shape[0]
        w=img.shape[1]
    jsdata={'imageWidth':w,'imageHeight':h,"version": "5.0.1",
    "flags": {}}
    ptsArr=[]
    for i in range(len(ndxlst)):
        assert(len(nptlst[i])==1)       
        for j in range(len(nptlst[i])):
            data={'group_id':groupids[i],'label':ndxlst[i],'flags':{},"shape_type": "polygon"}
            data['points']=nptlst[i][j]
            ptsArr.append(data)
    jsdata['shapes']=ptsArr
    if imgfile is not None:
        jsdata=ehl.AddLabelmeTailImageData(imgfile,jsdata)
    dio.writejsondictFormatFile(jsdata,jsfile)
def ConvertIntToMask(msk):
    maskint=np.zeros(msk.shape[:2],np.int32)
    for i in range(msk.shape[-1]):
        maskint[msk[...,i]>0]=i
    return maskint
def GetConnectAreaMask(msk,classes):
    h,w=msk.shape[:2]
    mskint=ConvertIntToMask(msk)
    kernalInt = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    instmsk=np.zeros_like(mskint).astype(np.int32)
    instlst=[[]]
    for i in range(msk.shape[2]):
        curmsk=msk[...,i]
        curmsk*=255
        curmsk=curmsk.astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(curmsk, connectivity=8)
        for j in range(1,num_labels):
            areamsk=labels==j
            diamsk=np.zeros(areamsk.shape,np.uint8)
            diamsk[areamsk]=255
            newmsk = cv2.dilate(diamsk, kernalInt, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            deltamsk=newmsk-diamsk
            adjoin=mskint[deltamsk>0]
            adjoin=np.unique(adjoin).tolist()
            adjointxt=[classes[adj] for adj in adjoin]
            dex=len(instlst)
            instmsk[areamsk]=dex
            instlst.append([i,adjoin,adjointxt])
    return instmsk,instlst
def GetConnectAreaInt(mskint,classes):
    msk=ConvertINToMaskClsNum(mskint,len(classes))
    return GetConnectAreaMask(msk,classes)