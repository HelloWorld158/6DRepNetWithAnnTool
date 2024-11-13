import BasicUseFunc as basFunc
import os, time, cv2, sys, math 
import numpy as np
import time, datetime
import os, random
from PIL import Image 
import matplotlib.image as matimage 
import imgaug as ia
import imgaug.augmenters as iaa
import DataIO as dio
import PIL as pil
import math
from typing import List,Union,Tuple,Any
######################################################
#######特别注意，opencv的数组已经与numpy合二为一了，现在
#######不需要额外再转换opencv的数组函数下面的某些函数会
#######废弃###########################################
######################################################
def load_image(path):
    timg=Image.open(path)
    timg=timg.convert("RGB")
    timg=np.array(timg,dtype=np.uint8)
    image=timg
    return image
def save_image(path,image):
    image.save(path)
################已经废弃，现在不再需要转换函数了################
def CvImageConvertNumpy(image):
    return np.array(image,dtype=np.uint8)
################已经废弃，现在不再需要转换函数了################
def NumpyConvertCvImage(ndarray):    
    ndarray = cv2.cvtColor(ndarray,cv2.COLOR_RGB2BGR)
    return ndarray
def random_crop_labelimage(image, label, crop_height, crop_width):
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        raise Exception('Image and label must have the same dimensions!')

    if (crop_width <= image.shape[1]) and (crop_height <= image.shape[0]):
        x = random.randint(0, image.shape[1] - crop_width)
        y = random.randint(0, image.shape[0] - crop_height)

        if len(label.shape) == 3:
            return image[y:y + crop_height, x:x + crop_width, :], label[y:y + crop_height, x:x + crop_width, :]
        else:
            return image[y:y + crop_height, x:x + crop_width, :], label[y:y + crop_height, x:x + crop_width]
    else:
        raise Exception('Crop shape (%d, %d) exceeds image dimensions (%d, %d)!' % (
        crop_height, crop_width, image.shape[0], image.shape[1]))
    return None,None
def CropNumpy_labelImage(image,label,box):#left,top,right,bottom'
    nbox=[]
    for i in range(2):
        b=min(max(0,box[2*i]),image.shape[1])
        t=min(max(0,box[2*i+1]),image.shape[0])
        nbox.append(b)
        nbox.append(t)
    image=image[nbox[1]:nbox[3],nbox[0]:nbox[2],:]
    label=label[nbox[1]:nbox[3],nbox[0]:nbox[2],:]
    return image,label
def CropNumpy_Image(image,box):#left,top,right,bottom
    nbox=[]
    for i in range(2):
        b=min(max(0,box[2*i]),image.shape[1])
        t=min(max(0,box[2*i+1]),image.shape[0])
        nbox.append(b)
        nbox.append(t)
    image=image[nbox[1]:nbox[3],nbox[0]:nbox[2],:]
    return image
#水平1   垂直0
def flip_labelimage(label,image,hvflip):
    image=NumpyConvertCvImage(image)
    label=NumpyConvertCvImage(label)
    label=cv2.flip(label,hvflip)
    image=cv2.flip(image,hvflip)
    image=CvImageConvertNumpy(image)
    label=CvImageConvertNumpy(label)
    return image,label
def ControlRandBright(image,brightness):
    image=NumpyConvertCvImage(image)
    factor = 1.0 + random.uniform(-1.0 * args.brightness, args.brightness)  # random.uniform[a,b]随机生成一个a,b之间的数
    # table = np.array([((i / 255.0) * factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    table = np.array([((i / 255.0) * factor) * 255 for i in np.arange(0, 48)]).astype(np.uint8)
    image = cv2.LUT(image, table)
    image=CvImageConvertNumpy(image)
    return image
def Rotate_ImageLabel(image,label,angle,pos=None):
    if(not pos):
        pos=[image.shape[1]/2,image.shape[0]/2]
    M = cv2.getRotationMatrix2D((pos[0],pos[1]), angle, 1.0)
    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
                                     flags=cv2.INTER_LINEAR)
    label = cv2.warpAffine(label, M, (label.shape[1], label.shape[0]),
                                      flags=cv2.INTER_NEAREST)    
    return image,label
def Translate_ImageLabel(image,label,tx,ty):
    M= np.float32([[1, 0, tx], [0, 1, ty]])
    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
                                     flags=cv2.INTER_LINEAR)
    label = cv2.warpAffine(label, M, (label.shape[1], label.shape[0]),
                                      flags=cv2.INTER_NEAREST)
    return image,label
def Resize_ImageLabel(image,label,scale,pos=None):
    size=image.shape
    nsize=[]
    for i in range(2):
        nsize.append(int(size[i]*scale))
    oimage=cv2.resize(image,tuple(nsize),interpolation=cv2.INTER_LINEAR)
    olabel=cv2.resize(label,tuple(nsize),interpolation=cv2.INTER_NEAREST)
    npimg=np.zeros(image.shape,dtype=np.uint8)
    nplabel=np.zeros(label.shape,dtype=np.uint8)
    imgshape=[int(npimg.shape[0]/2-oimage.shape[0]/2),int(npimg.shape[0]/2+oimage.shape[0]/2),int(npimg.shape[1]/2-oimage.shape[1]/2)
    ,int(npimg.shape[1]/2+oimage.shape[1]/2)]
    npimg[imgshape[0]:imgshape[1],imgshape[2]:imgshape[3],:]=oimage[0:(imgshape[1]-imgshape[0]),0:(imgshape[3]-imgshape[2]),:]
    nplabel[imgshape[0]:imgshape[1],imgshape[2]:imgshape[3],:]=olabel[0:(imgshape[1]-imgshape[0]),0:(imgshape[3]-imgshape[2]),:]
    npimg=np.asarray(npimg,dtype=np.uint8)
    nplabel=np.asarray(nplabel,dtype=np.uint8)
    if(pos):
        npos=np.array(pos,np.float32)
        curpos=npos.copy()
        npos-=[npimg.shape[0]/2,npimg.shape[1]/2]
        npos*=scale
        npos+=[npimg.shape[0]/2,npimg.shape[1]/2]
        delta=curpos-npos
        Translate_ImageLabel(npimg,nplabel,delta[0],delta[1])
    return npimg,nplabel
def GetImagePolyMask(img,ptsArr):
    ptsArr=np.array(ptsArr,np.int32)
    nimg=np.zeros_like(img,np.uint8)
    cv2.fillPoly(nimg,[ptsArr],255)
    mask=nimg[:,:,0]>10
    return mask
#原始图像坐标=(新图像坐标-posLt)/scale

def GenerateExpandImageData(timg,tHeight=224,tWidth=224):
    timg=Image.fromarray(timg)
    nwHeight=tWidth
    nwWidth=tHeight
    scale=max(timg.size[0],timg.size[1])
    asize=np.asarray(timg.size,dtype=np.float32)
    asize/=np.float(scale)
    if((asize[1]/asize[0])>=(np.float(tHeight)/np.float(tWidth))):
        asize*=tHeight
    else:
        asize*=tWidth
    nsize=(int(asize[0]),int(asize[1]))
    aimg=timg.resize(nsize,pil.Image.ANTIALIAS)
    aimg=np.asarray(aimg,np.uint8)
    bimg=np.zeros([tHeight,tWidth,3],np.uint8)
    bimg[:aimg.shape[0],:aimg.shape[1],:]=aimg
    picCenter=np.array([bimg.shape[0]/2.0,bimg.shape[1]/2.0],np.float32)
    center=np.array([aimg.shape[0]/2.0,aimg.shape[1]/2.0],np.float32)
    picCenter-=center
    for i in range(picCenter.shape[0]):
        picCenter[i]/=bimg.shape[i]
    aug=iaa.Affine(translate_percent={"x": (picCenter[1],picCenter[1]), "y": (picCenter[0], picCenter[0])})
    bimg=aug.augment_image(bimg)
    return bimg
#原始图像坐标=(新图像坐标-posLt)/scale
#这个函数可以把trans,scale 返回到调用者中，变换是从timg变换到bimg
#要想进行逆变换需要添加下面的代码
#trans=-trans
#scale=1.0/scale
#boxes=ConvertBoxesTranScale(boxes,trans,scale)
#可以将bimg的boxes转换为原先的timg boxes
#要想进行box 从timg->bimg变换使用下面函数
#boxes=ConvertBoxesScaleTran(boxes,scale,trans)
def GenerateExpandImageDataExFunc(timg,tHeight=224,tWidth=224):
    timg=Image.fromarray(timg)
    nwHeight=tWidth
    nwWidth=tHeight
    scale=float(max(timg.size[0],timg.size[1]))
    asize=np.asarray(timg.size,dtype=np.float32)
    asize/=np.float(scale)
    if((asize[1]/asize[0])>=(np.float(tHeight)/np.float(tWidth))):
        sscale=float(tHeight)/scale
        asize*=tHeight
    else:
        sscale=float(tWidth)/scale
        asize*=tWidth
    nsize=(int(asize[0]),int(asize[1]))
    aimg=timg.resize(nsize,pil.Image.ANTIALIAS)
    aimg=np.asarray(aimg,np.uint8)
    bimg=np.zeros([tHeight,tWidth,3],np.uint8)
    bimg[:aimg.shape[0],:aimg.shape[1],:]=aimg
    picCenter=np.array([bimg.shape[0]/2.0,bimg.shape[1]/2.0],np.float32)
    center=np.array([aimg.shape[0]/2.0,aimg.shape[1]/2.0],np.float32)
    picCenter-=center
    otrans=picCenter.copy()
    for i in range(picCenter.shape[0]):
        picCenter[i]/=bimg.shape[i]
    aug=iaa.Affine(translate_percent={"x": (picCenter[1],picCenter[1]), "y": (picCenter[0], picCenter[0])})
    bimg=aug.augment_image(bimg)
    trans=np.zeros_like(picCenter)
    trans[1]=otrans[0]
    trans[0]=otrans[1]
    #bimg,trans,scale(均是从timg转换到bimg的矩阵)
    return bimg,trans,sscale
#Y=scale*(x+trans)
#注意仅支持ltrb
def ConvertBoxesTranScale(boxes,trans,scale):
    for i in range(len(boxes)//2):
        boxes[2*i]=scale*(boxes[2*i]+trans[0])
        boxes[2*i+1]=scale*(boxes[2*i+1]+trans[1])
    return boxes
def ConvertBoxesScaleTran(boxes,scale,trans):
    for i in range(len(boxes)//2):
        boxes[2*i]=scale*(boxes[2*i])+trans[0]
        boxes[2*i+1]=scale*(boxes[2*i+1])+trans[1]
    return boxes
#images_aug = seq.augment_images(images)#images是4维NHWC的数组
#image_aug = rotate.augment_image(image) image 是三维HWC的数组
def ImgaugAugument():
     
    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
     
    # Define our sequence of augmentation steps that will be applied to every image
    # All augmenters with per_channel=0.5 will sample one value _per image_
    # in 50% of all cases. In all other cases they will sample new values
    # _per channel_.
    #iaa.OneOf([iaa.GaussianBlur((0, 0.8))]),
    #                iaa.Sharpen(alpha=(0, 0.08), lightness=(0.97, 1.03)), # sharpen images
    #                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
    #                iaa.Add((-5, 5), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
    #                iaa.AddToHueAndSaturation((-1, 1)), 
    #                iaa.ContrastNormalization((0.6, 1.0), per_channel=0.5), # improve or worsen the contrast
    #                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5), # horizontally flip 50% of all images左右翻转
            iaa.Flipud(0.2), # vertically flip 20% of all images上下翻转
            # crop images by -5% to 10% of their height/width
            sometimes(iaa.CropAndPad(
                percent=(-0.05, 0.1),
                pad_mode=ia.ALL,
                pad_cval=(0, 255)
            )),#随机剪裁
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45), # rotate by -45 to +45 degrees
                shear=(-16, 16), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                #mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)这个一般别设置，否则会出现别的地方都有图像
            )),#仿射变换
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                [
                    #sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # 部分图像超像素，部分图像模糊,毛玻璃特效要小心使用
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0Gauss模糊
                        iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                    #iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                    # search either for all edges or for directed edges,
                    # blend the result with the original image using a blobby mask
                    #iaa.SimplexNoiseAlpha(iaa.OneOf([
                    #    iaa.EdgeDetect(alpha=(0.5, 1.0)),
                    #    iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                    #])),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                    #iaa.OneOf([
                    #    iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                    #    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                    #]),
                    #iaa.Invert(0.05, per_channel=True), # invert color channels
                    iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                    iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                    # either change the brightness of the whole image (sometimes
                    # per channel) or change the brightness of subareas
                    iaa.OneOf([
                        iaa.Multiply((0.5, 1.5), per_channel=0.5),
                        iaa.FrequencyNoiseAlpha(
                            exponent=(-4, 0),
                            first=iaa.Multiply((0.5, 1.5), per_channel=True),
                            second=iaa.ContrastNormalization((0.5, 2.0))
                        )
                    ]),
                    iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                    #iaa.Grayscale(alpha=(0.0, 1.0)),#灰度失真，有可能大片像素都是一个颜色
                    #sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)#毛玻璃特效2，注意小心用
                    #sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around#扭曲特效，要小心使用
                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                ],
                random_order=True
            )
        ],
        random_order=True
    )
    return seq
def GetImageWindow(npimg,x,y,sepw,seph):
    l,t,r,b=x,y,x+sepw,y+seph
    box=[l,t,r,b]
    for i in range(2):
        box[2*i]=max(0,min(npimg.shape[1],box[2*i]))
        box[2*i+1]=max(0,min(npimg.shape[0],box[2*i+1]))
    return npimg[box[1]:box[3],box[0]:box[2],:]
def ConcatTwoImageScale(imgA,imgB,scale):
    imga=imgA
    imgb=imgB
    sizea=[int(imga.height*scale),int(imga.width*scale)]
    sizeb=[int(imgb.height*scale),int(imgb.width*scale)]
    boxesA=(np.array(boxesA)*scale).tolist()
    boxesB=(np.array(boxesB)*scale).tolist()
    height=sizea[0]+sizeb[0]
    width=max(sizea[1],sizeb[1])
    imga=imga.resize([sizea[1],sizea[0]],pil.Image.BILINEAR)
    imgb=imgb.resize([sizeb[1],sizeb[0]],pil.Image.BILINEAR)
    img=np.zeros([height,width,3],np.uint8)
    m=np.array(imga,np.uint8)
    img[:sizea[0],:sizea[1],:]=m
    m=np.array(imgb,np.uint8)
    img[sizea[0]:sizea[0]+sizeb[0],:sizeb[1],:]=m
    return img
def ChangeColorImage(orimg,mean,std):
    orimg=np.array(orimg,np.float32)
    orimean=np.mean(orimg.reshape([-1,3]),0)
    oristd=np.std(orimg.reshape([-1,3]),0)
    newimg=orimg.copy()
    newimg-=orimean
    newimg/=oristd
    newimg*=std
    newimg+=mean
    newimg=np.where(newimg>255,255,newimg)
    newimg=np.where(newimg<0,0,newimg)
    return newimg
def ChangeMaskColorImage(img,maskPos,newimg,newmskPos):
    curimg=newimg[newmskPos[0],newmskPos[1]]
    nimgmean=np.mean(curimg,0)
    nimgstd=np.std(curimg,0)
    img=ChangeColorImage(img,nimgmean,nimgstd)
    newimg[newmskPos[0],newmskPos[1]]=img[maskPos[0],maskPos[1]]
    #newimg=GivePosImage(img,maskPos,newimg,newmskPos)
    return newimg
def GivePosImage(orimg,oripos,newimg,newpos):
    if(orimg.shape[-1]!=newimg.shape[-1]):
        print('pos not matched')
        exit()
    result=[]
    for i in range(orimg.shape[-1]):
        a=orimg[:,:,i]
        b=newimg[...,i]
        b[newpos]=a[newpos]
        b=np.expand_dims(b,len(b.shape))
        result.append(b)
    result=np.concatenate(result,-1)
    return result
def GiveImageMask(orimg,mask,newimg,newmask,dex,left,top,keepColor):
    import BasicCocoFunc as bascoco
    import BasicAlgorismGeo as basGeo
    msk=basGeo.MaskConvert(mask)
    box=basGeo.extract_bboxes(msk)[0]
    xywh=basGeo.LtRbToxywh(box.tolist())
    mskpos=np.where(mask)
    mskpos=np.array(mskpos,np.int32)
    mskpos=np.transpose(mskpos,[1,0])
    nmskpos=mskpos-np.array([xywh[1],xywh[0]],np.int32)+np.array([top,left],np.int32)
    ftpos=np.min(nmskpos,-1)
    ftpos=np.expand_dims(ftpos,len(ftpos.shape))
    ftpos=np.concatenate([ftpos,ftpos],-1)
    posdex=np.where(ftpos>=0)
    if(posdex[0].shape[0]==0):
        return newimg,newmask
    nmskpos=nmskpos[posdex].reshape([-1,2])
    mskpos=mskpos[posdex].reshape([-1,2])
    hpos,wpos=nmskpos[...,1],nmskpos[...,0]
    hpos=hpos<newimg.shape[1]
    wpos=wpos<newimg.shape[0]
    ftpos=np.expand_dims(np.logical_and(hpos,wpos),len(wpos.shape))
    ftpos=np.concatenate([ftpos,ftpos],-1)
    posdex=np.where(ftpos)
    if(posdex[0].shape[0]==0):
        return newimg,newmask
    nmskpos=nmskpos[posdex].reshape([-1,2]).transpose([1,0])
    mskpos=mskpos[posdex].reshape([-1,2]).transpose([1,0])
    if(keepColor):
        newimg=ChangeMaskColorImage(orimg,mskpos,newimg,nmskpos)
    else:
        newimg[nmskpos[0],nmskpos[1]]=orimg[mskpos[0],mskpos[1]]
        #newimg=GivePosImage(orimg,mskpos,newimg,nmskpos)
    newmask[nmskpos[0],nmskpos[1]]=dex
    return newimg,newmask
class GenOccupiedAug(object):
    def __init__(sf,files,filterClasses,randbox=2,keepColor=0.0,coverbox=5,boxmin=0.02,minObjs=10,outSide=20):
        sf.filedir,_,_=basFunc.GetfileDirNamefilter(files[0])
        sf.allfiles=files
        sf.filterClasses=filterClasses
        sf.InitSegFile()
        sf.outSide=outSide
        sf.randbox=randbox
        sf.keepColor=keepColor
        sf.coverbox=coverbox
        sf.boxMin=boxmin
        sf.minObjs=minObjs        
    def InitSegFile(sf):
        segFile=os.path.join(os.getcwd(),'segfile.json')
        filedir,_,_=basFunc.GetfileDirNamefilter(sf.allfiles[0])
        if(os.path.exists(segFile)):
            sf.segdata=dio.getjsondata(segFile)
        else:
            import BasicCocoFunc as bascoco
            import BasicAlgorismGeo as basGeo
            sf.segdata={'file':[],'seg':[],'class':[],'area':[]}
            for i in range(len(sf.allfiles)):
                basFunc.Process(i,len(sf.allfiles))
                #if(i>1000):break
                f=sf.allfiles[i]
                img=dio.GetOriginImageData(f)
                _,name,ftr=basFunc.GetfileDirNamefilter(f)
                jsonfile=os.path.join(filedir,name+'.json')
                data=dio.getjsondata(jsonfile)
                segs,clss=data['masks'],data['classes']
                for j in range(len(segs)):
                    mask=bascoco.ConvertSegToMask(img,segs[j])
                    mask=basGeo.MaskConvert(mask)
                    box=basGeo.extract_bboxes(mask)[0]
                    xywh=basGeo.LtRbToxywh(box.tolist())
                    area=xywh[2]*xywh[3]
                    if(area==0.0):continue
                    if(clss[j] not in sf.filterClasses):continue
                    sf.segdata['file'].append(f)
                    sf.segdata['seg'].append(segs[j])
                    sf.segdata['class'].append(clss[j])
                    sf.segdata['area'].append(area)
            allsort=zip(sf.segdata['file'],sf.segdata['seg'],sf.segdata['class'],sf.segdata['area'])
            m=list(allsort)
            allsorts=sorted(m,key=lambda x:x[3])
            sf.segdata['file'],sf.segdata['seg'],sf.segdata['class'],sf.segdata['area']=zip(*allsorts)
            dio.writejsondictFilelines(sf.segdata,os.path.join(os.getcwd(),'segfile.json'))    
    def FindNearestSeg(sf,area,nearnum=20):
        nearest=-1
        neardex=-1
        seglist=[]
        randseg=sf.randbox
        areas=np.array(sf.segdata['area'],np.float32)
        deltaarea=np.abs(areas-area)
        mindex=np.argmin(deltaarea)
        dexlst=[np.random.randint(max(0,mindex-nearnum),min(len(sf.segdata['area']),mindex+nearnum)) for i in range(1)]
        return dexlst      
    def GenerateRandPos(sf,newmask,curDex):
        pos=[]
        curMask=newmask==curDex
        img=np.zeros_like(curMask,np.uint8)
        img[curMask]=255   
        kernel=np.ones([2*sf.outSide+1,2*sf.outSide+1])
        #matimage.imsave('test.jpg',img)
        img=cv2.dilate(img,kernel)
        #matimage.imsave('testa.jpg',img)
        img=cv2.Canny(img,threshold1=100,threshold2=150)
        #matimage.imsave('testb.jpg',img)
        kernel=np.ones([2*sf.coverbox+1,2*sf.coverbox+1])
        img=cv2.dilate(img,kernel)
        #matimage.imsave('testc.jpg',img)
        imgmask=img>128
        zeromask=curMask!=0
        othermask=curMask!=curDex
        mergemask=np.logical_and(zeromask,othermask)
        mergemask=np.logical_not(mergemask)
        finalmask=np.logical_and(mergemask,imgmask)
        #matimage.imsave('testb.jpg',img)
        xypos=np.where(finalmask)
        xypos=[xypos[1],xypos[0]]
        xypos=np.array(xypos,np.int32)
        if(xypos.shape[1]==0):
            return None
        xypos=np.transpose(xypos,[1,0])
        pos=[xypos[np.random.randint(0,xypos.shape[0])] for i in range(1)]
        return pos
    def FilterMaskBox(sf,newimg,newmask,classes,start,end,sepcnt):
        import BasicAlgorismGeo as basGeo
        newmasks=basGeo.MaskNumConvert(newmask,start,end)
        boxes=basGeo.extract_bboxes(newmasks).tolist()
        finalbox=[]
        finalcls=[]
        for i in range(len(boxes)):
            b=boxes[i]
            m=basGeo.LtRbToxywh(b)
            mwh=min(m[2]/newimg.shape[1],m[3]/newimg.shape[0])
            if(mwh<sf.boxMin and i<sepcnt): continue            
            finalbox.append(b)
            finalcls.append(classes[i])
        return finalbox,finalcls
    def GenerateOccupiedObject(sf,file):
        import BasicCocoFunc as bascoco
        import BasicAlgorismGeo as basGeo
        orimg=dio.GetOriginImageData(file)
        d,name,ftr=basFunc.GetfileDirNamefilter(file)
        jsfile=os.path.join(d,name+'.json')
        dct=dio.getjsondata(jsfile)
        orisegs,oriclss=dct['masks'],dct['classes']
        newimg=orimg.copy()
        newmask=np.zeros(newimg.shape[:2],np.int32)
        maskcnt=1
        boxes=[]
        classes=[]
        oriareas=[]
        orisegss=[]
        for i in range(len(orisegs)):
            seg=orisegs[i]
            clss=oriclss[i]
            mask=bascoco.ConvertSegToMask(orimg,seg)
            cmask=basGeo.MaskConvert(mask)
            box=basGeo.extract_bboxes(cmask)[0]
            xywh=basGeo.LtRbToxywh(box.tolist())
            area=xywh[2]*xywh[3]
            if(area==0.0):continue
            oriareas.append(area)
            newmask[mask]=maskcnt
            orisegss.append(seg)
            classes.append(oriclss[i])
            maskcnt+=1
        sepcnt=maskcnt
        if(maskcnt<=sf.minObjs):
            for i in range(1,sepcnt):  
                pixsum=int(np.sum(newmask==i))
                #print(pixsum)
                #sys.stdout.flush()
                if(classes[i-1] not in sf.filterClasses):continue
                if(pixsum<300):continue
                for j in range(sf.randbox):
                    allpos=sf.GenerateRandPos(newmask,i)
                    if(not isinstance(allpos,list)):
                        return None,None,None
                    dexlst=sf.FindNearestSeg(oriareas[i-1])
                    dex=dexlst[0]
                    left,top=allpos[0][0],allpos[0][1]
                    addseg=sf.segdata['seg'][dex]
                    addfile=sf.segdata['file'][dex]
                    addimg=dio.GetOriginImageData(addfile)
                    addmask=bascoco.ConvertSegToMask(addimg,addseg)                
                    addcls=sf.segdata['class'][dex]                
                    newimg,newmask=GiveImageMask(addimg,addmask,newimg,newmask,maskcnt,left,top,False)
                    #matimage.imsave('test.jpg',newimg)
                    classes.append(addcls)
                    maskcnt+=1
            if(maskcnt==sepcnt):
                return None,None,None
            for i in range(len(orisegss)):
                seg=orisegss[i]
                mask=bascoco.ConvertSegToMask(orimg,seg)
                pos=np.where(mask)
                newimg[pos]=orimg[pos]
                newmask[mask]=i+1
        else:
            return None,None,None
        boxes,classes=sf.FilterMaskBox(newimg,newmask,classes,1,maskcnt,sepcnt)
        return newimg,boxes,classes
dctConfig=0
def GenerateOccupiedDataAug(dir,debugDir,txtFile,maxcnt=-1):
    files=basFunc.getdatas(dir,'*.jpg')
    import DataIO as dio
    import BasicAlgorismGeo as basGeo
    import BasicDrawing as basDraw
    allfile=[]
    count=0
    print('Judge File')
    nclasses=['person','bicycle','car','motorbike','bus','truck']
    for file in files:
        count+=1
        basFunc.Process(count,len(files))
        d,name,ftr=basFunc.GetfileDirNamefilter(file)
        jsfile=os.path.join(d,name+'.json')
        if(not os.path.exists(jsfile)):continue
        dct=dio.getjsondata(jsfile)
        if('masks' not in dct):continue
        allfile.append(file)
        #if(count>1000):break
    print('\nGenerate File')
    count=0
    gOAug=GenOccupiedAug(allfile,['bicycle'])
    if(maxcnt<0):
        maxcnt=int(len(allfile)*1.0)
    for i in range(maxcnt):
        file=allfile[np.random.randint(0,len(allfile))]
        count+=1
        if(count>maxcnt):break
        basFunc.Process(count,len(allfile))
        newimg,boxes,classes=gOAug.GenerateOccupiedObject(file)
        if(not isinstance(newimg,np.ndarray)):
            count-=1
            continue
        global dctConfig
        dctConfig['count']+=1
        dio.writejsondictFormatFile(dctConfig,configFile)   
        trainimg=os.path.join(dir,str(dctConfig['count'])+'.jpg')
        matimage.imsave(trainimg,newimg)
        trainlabel=os.path.join(dir,str(dctConfig['count'])+'.txt')
        fp=open(trainTxt,'a')
        fp.write(trainimg+'\n')
        fp.close()
        fp=open(trainlabel,'w')
        for i in range(len(classes)):
            dex=nclasses.index(classes[i])
            box=boxes[i]
            box=basGeo.LtRbToxywh(box)
            for j in range(2):
                box[j*2]/=float(newimg.shape[1])
                box[j*2+1]/=float(newimg.shape[0])
            fp.write(str(dex)+" "+" ".join([str(b) for b in box])+'\n')
        fp.close()     
        debugimg=os.path.join(debugDir,str(dctConfig['count'])+'.jpg')  
        if(os.path.exists(debugimg)):
            os.remove(debugimg)
        basDraw.DrawImageCopyFileRectangles(debugimg,trainimg,boxes,namelst=classes)
def GetDict():
    dct={'count':0}
    return dct
#这个类不要初始化，不要实例化，仅作为通用函数的集合体
class SiftDeal(object):
    def sift_kp(image):
        gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # sift = cv2.xfeatures2d.SURF_create(400)
        sift=cv2.xfeatures2d.SIFT_create()
        kp,des = sift.detectAndCompute(image,None)
        kp_image = cv2.drawKeypoints(gray_image,kp,None)
        return kp_image,kp,des
    def sift_Match(des1,des2):
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            #  配置参数 0.8
            if m.distance < 0.8 * n.distance:
                good.append(m)
        return good
    def cal_point(x, y, M):
        if(not M):return x,y
        x0, y0 = (M[0,0]*x+M[0,1]*y+M[0,2])/(M[2,0]*x+M[2,1]*y+M[2,2]), (M[1,0]*x+M[1,1]*y+M[1,2])/(M[2,0]*x+M[2,1]*y+M[2,2])
        return x0, y0            
    def siftImageAlignment(img1,img2):
        _,kp1,des1 = SiftDeal.sift_kp(img1)
        _,kp2,des2 = SiftDeal.sift_kp(img2)
        goodMatch = SiftDeal.sift_Match(des1,des2)
        if len(goodMatch) > 4:
            ptsA= np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
            ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
            ransacReprojThreshold = 4
            H, status =cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold)
            return H
        return None
    def CaculatePoints(poses,M):
        if(not M):
            return poses
        res=[]
        for i in range(len(poses)):
            pos=poses[i]
            x,y=SiftDeal.cal_point(pos[0],pos[1],M)
            res.append([x,y])
        return res
def ArrangeImages(imgs:List[np.ndarray],rows:int=None):
    if rows is None:
        rows=math.ceil(math.sqrt(len(imgs)))
    cols=(len(imgs)+rows-1)//rows
    imglst=[]
    for i in range(cols):
        imgexs=[]
        for j in range(rows):
            if i*rows+j<len(imgs):
                imgexs.append(imgs[i*rows+j])
            else:
                imgexs.append(np.zeros_like(imgs[0]))
        imgex=np.concatenate(imgexs,1)
        imglst.append(imgex)
    imglst=np.concatenate(imglst,0)
    return imglst
'''
size:(w,h)
'''
def ArrangeImages2FileMMCV(imgs:Union[str,List[np.ndarray],List[str],np.ndarray],
                           outfile:str,
                           size:Tuple[int,int]=None,
                           sepdex:int=1,
                           rows:int=None,
                           filter:str='*.jpg'):
    import mmcv
    if type(imgs) is str:
        files=basFunc.getdatas(imgs,filter)
        imgs=files
    if type(imgs) is np.ndarray:
        lst=[imgs[i] for i in range(imgs.shape[0])]
        imgs=lst
    imglst=[]
    for i,img in enumerate(imgs):
        basFunc.Process(i,len(imgs))
        if i%sepdex!=0:continue
        if type(img) is str:
            curimg=dio.GetOriginImageData(img)
        else:
            curimg=img
        if size is None:
            curimg=img
        else:
            tpimg=mmcv.imrescale(curimg,size)
            if i==0:print(tpimg.shape)
            curimg=np.zeros(list(size)[::-1]+[3],np.uint8)
            curimg[:tpimg.shape[0],:tpimg.shape[1]]=tpimg
        imglst.append(curimg)
    imgfinal=ArrangeImages(imglst,rows)
    matimage.imsave(outfile,imgfinal)
def MaxMinImage(img,epsion=1e-10,clip=None):
    if clip is not None:
        img=np.clip(img,clip[0],clip[1])
    if len(img.shape)==3:
        mximg=img.max(0,keepdims=True).max(1,keepdims=True)
        mnimg=img.min(0,keepdims=True).min(1,keepdims=True)
    else:
        mximg=img.max(1,keepdims=True).max(2,keepdims=True)
        mnimg=img.min(1,keepdims=True).min(2,keepdims=True)
    outimg=(img-mnimg)/(mximg-mnimg+epsion)
    outimg=np.clip(outimg,0.0,1.0)
    outimg=outimg*255.0
    outimg=outimg.astype(np.uint8)
    return outimg
def DrawTensorNdarrayMMCV(img:List[Any],filedir:str
                   ,usedir:bool=False,clip:List[float]=None,
                   singleChannle:bool=False,rows:int=None,
                   size:Tuple[int,int]=None,
                   dims:List[int]=[0,2,3,1])->np.ndarray:
    from torch import Tensor
    import mmcv
    if type(img) is Tensor:
        img=img.detach().cpu().numpy()
    img=np.transpose(img,dims)
    if singleChannle:
        img=np.transpose(img,[0,3,1,2])
        img=img.reshape([-1,img.shape[2],img.shape[3],1])
    if usedir:
        basFunc.MakeEmptyDir(filedir)
        outimg=[]
        for i in range(img.shape[0]):
            curimg=img[i]
            curfile=os.path.join(filedir,str(i)+'.jpg')
            curimg=MaxMinImage(curimg,clip=clip)
            if curimg.shape[-1]==1:
                curimg=cv2.applyColorMap(curimg,cv2.COLORMAP_JET)
            matimage.imsave(curfile,curimg)
            outimg.append(curimg)
        outimg=np.stack(outimg,0)
    else:
        curimg=MaxMinImage(img,clip=clip)
        imgs=[]
        for i in range(curimg.shape[0]):
            img=curimg[i]
            if img.shape[-1]==1:
                img=cv2.applyColorMap(img,cv2.COLORMAP_JET)
            if size is not None:
                img=mmcv.imrescale(img,size)
                if i==0:print(img.shape)
                zimg=np.zeros(list(size)[::-1]+[img.shape[-1]],np.uint8)
                zimg[:img.shape[0],:img.shape[1]]=img
                img=zimg
            imgs.append(img)
        curimg=np.stack(imgs,0)
        curimg=ArrangeImages(curimg,rows)
        matimage.imsave(filedir,curimg)
        outimg=curimg
    return outimg
if(__name__=="__main__"):
    import DataIO as dio
    [trainDir,validDir,trainTxt,validTxt,weightDir,configFile,debugDir]\
    =basFunc.GetCurDirNames\
    (['train','valid','train.txt','test.txt','weight','config.json','debugDir'])
    dctConfig=dio.InitJsonConfig(GetDict,configFile)
    GenerateOccupiedDataAug(trainDir,debugDir,trainTxt)
    exit()
    picPath='D:/data/CityControl/MRCNN/train_dataset/img/img_5.png'
    labelPath='D:/data/CityControl/MRCNN/train_dataset/label/label_5.png'
    image=load_image(picPath)
    label=load_image(labelPath)
    import BasicUseFunc as basFunc
    respath=os.path.join(os.getcwd(),'result')
    basFunc.MakeEmptyDir(respath)
    npicPath=os.path.join(respath,'image.png')
    nlabelPath=os.path.join(respath,'label.png')
    timg,poslt,scale=ResizeCenterNewPic(image,320,480)
    matimage.imsave(npicPath,timg)
    exit()
    image,label=Resize_ImageLabel(image,label,0.5)
    matimage.imsave(npicPath,image)
    matimage.imsave(nlabelPath,label)
    image,label=Rotate_ImageLabel(image,label,30)
    matimage.imsave(npicPath,image)
    matimage.imsave(nlabelPath,label)
    image,label=Translate_ImageLabel(image,label,-50,-50)
    matimage.imsave(npicPath,image)
    matimage.imsave(nlabelPath,label)
    

    
    

