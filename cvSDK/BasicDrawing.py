import BasicUseFunc as basFunc
import numpy as np
import os
import sys
import glob
import struct
import matplotlib.image as imgUse
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import BasicAlgorismGeo as basAlgo
import shutil
from PIL import Image,ImageDraw,ImageFont
from PIL import ImageFile
import cv2
ImageFile.LOAD_TRUNCATED_IMAGES = True
fontpath=os.path.join(os.path.dirname(__file__),'MSYHBD.ttc')
fontSize=9
def ChangeFontSize(size):
    global fontSize
    fontSize=size
def GetCurFont():
    return ImageFont.truetype(fontpath, fontSize)
#fill="#879f65"
def DecorateNumpy(func):
    def newfunc(*args,useNpy=False,**kwargs):
        if useNpy:
            if 'img' in kwargs:
                kwargs['img']=Image.fromarray(kwargs['img'])
            else:
                args=list(args)
                args[0]=Image.fromarray(args[0])
            img=func(*args,**kwargs)
            img=np.array(img,np.uint8)
            return img
        else:
            return func(*args,**kwargs)
    return newfunc
@DecorateNumpy            
def DrawImageRectangles(img,ltrblst,color='white',namelst=None,namedelta=3,outline=0,font=ImageFont.truetype(fontpath, fontSize)): 
    idraw=ImageDraw.Draw(img)
    for i in range(len(ltrblst)):
        m=ltrblst[i]
        idraw.rectangle((m[0]-outline,m[1]-outline,m[2]+outline,m[3]+outline),width=2,outline=color)
        if(namelst):
            if(type(namedelta) is list or type(namedelta) is tuple):
                idraw.text((m[0]+namedelta[0],m[1]+namedelta[1]),namelst[i],fill=color,font=font)
            else:
                idraw.text((m[0]+namedelta,m[1]+namedelta),namelst[i],fill=color,font=font)
    return img
@DecorateNumpy
def DrawImageRectanglesFixText(img,ltrblst,color='white',namelst=None,namedelta=3,elsedelta=[3,3],outline=0,font=ImageFont.truetype(fontpath, fontSize)): 
    idraw=ImageDraw.Draw(img)
    for i in range(len(ltrblst)):
        m=ltrblst[i]
        idraw.rectangle((m[0]-outline,m[1]-outline,m[2]+outline,m[3]+outline),width=2,outline=color)
        if(namelst):
            if(type(namedelta) is list or type(namedelta) is tuple):
                x=m[0]+namedelta[0]
                y=m[1]+namedelta[1]
                if(x<0 or y<0):
                    x=m[0]+elsedelta[0]
                    y=m[1]+elsedelta[1]
                idraw.text((x,y),namelst[i],fill=color,font=font)
            else:
                idraw.text((m[0]+namedelta,m[1]+namedelta),namelst[i],fill=color,font=font)
    return img
@DecorateNumpy
def DrawImageCenterBoxesFixText(img,cxtlst,color='white',namelst=None,namedelta=3,elsedelta=[3,3],outline=0,font=ImageFont.truetype(fontpath, fontSize)): 
    ltrblst=[]
    for m in cxtlst:
        r=basAlgo.XywhToLtRb(m)
        ltrblst.append(r)
    return DrawImageRectanglesFixText(img,ltrblst,color,namelst,namedelta,elsedelta,outline,font)
def DrawImageRectanglesToFile(img,svfile,ltrblst,color='white',namelst=None,namedelta=3,outline=0,font=ImageFont.truetype(fontpath, fontSize)):
    img=Image.fromarray(img)
    DrawImageRectangles(img,ltrblst,color,namelst,namedelta,outline,font)
    t=ConvertImageAsArray(img)
    imgUse.imsave(svfile,t)
#img=img.convert('RGB')
def ConvertImageAsArray(img):
    t=np.asarray(img)
    t=np.require(t,requirements=['O','W'])
    t.setflags(write=1)
    return t
def ConvertImageAsFloatArray(img):
    t=np.asarray(img,dtype=np.float32)
    t=np.require(t,requirements=['O','W'])
    t.setflags(write=1)
    t/=255.0
    return t
@DecorateNumpy
def DrawImageCenterBoxes(img,cxtlst,color='white',namelst=None,namedelta=2,outline=0,font=ImageFont.truetype(fontpath, fontSize)):
    ltrblst=[]
    for m in cxtlst:
        r=basAlgo.XywhToLtRb(m)
        ltrblst.append(r)
    return DrawImageRectangles(img,ltrblst,color,namelst,namedelta,outline,font)
def DrawImageCenterBoxesToFile(img,svfile,cxtlst,color='white',namelst=None,namedelta=2,outline=0,font=ImageFont.truetype(fontpath, fontSize)):
    ltrblst=[]
    for m in cxtlst:
        r=basAlgo.XywhToLtRb(m)
        ltrblst.append(r)
    DrawImageRectanglesToFile(img,svfile,ltrblst,color,namelst,namedelta,outline,font)
def DrawImageFileRectangles(imgfile,svfile,ltrblst,color='white',namelst=None,namedelta=2,outline=0,font=ImageFont.truetype(fontpath, fontSize)):
    img=Image.open(imgfile)
    DrawImageRectangles(img,ltrblst,color,namelst,namedelta,outline,font)
    t=ConvertImageAsArray(img)
    imgUse.imsave(svfile,t)
def DrawImageFileCenterBoxes(imgfile,svfile,cxtlst,color='white',namelst=None,namedelta=2,outline=0,font=ImageFont.truetype(fontpath, fontSize)):
    ltrblst=[]
    for m in cxtlst:
        r=basAlgo.XywhToLtRb(m)
        ltrblst.append(r)
    DrawImageFileRectangles(imgfile,svfile,ltrblst,color,namelst,namedelta,outline,font)
def DrawImageCopyFileRectangles(imgfile,srcfile,ltrblst,color='white',namelst=None,namedelta=2,outline=0,font=ImageFont.truetype(fontpath, fontSize)):
    if(not os.path.exists(imgfile)):
        shutil.copy(srcfile,imgfile)
    DrawImageFileRectangles(imgfile,imgfile,ltrblst,color,namelst,namedelta,outline,font)
def DrawImageCopyFileCenterBoxes(imgfile,srcfile,cxtlst,color='white',namelst=None,namedelta=2,outline=0,font=ImageFont.truetype(fontpath, fontSize)):
    if(not os.path.exists(imgfile)):
        shutil.copy(srcfile,imgfile)
    DrawImageFileCenterBoxes(imgfile,imgfile,cxtlst,color,namelst,namedelta,outline,font)
@DecorateNumpy
def DrawImageText(img,poslst,txtlst,color='#ffffff',font=ImageFont.truetype(fontpath, fontSize)):
    idraw= ImageDraw.Draw(img)
    for i in range(len(poslst)):
        idraw.text((poslst[i][0],poslst[i][1]),txtlst[i],fill=color,font=font)
    return img
def DrawImageTextToFile(img,svfile,poslst,txtlst,color='#ffffff',font=ImageFont.truetype(fontpath, fontSize)):
    img=Image.fromarray(img)
    DrawImageText(img,poslst,txtlst,color,font)
    t=ConvertImageAsArray(img)
    imgUse.imsave(svfile,t)
def DrawImageTextFile(oriImageFile,newImageFile,poslst,txtlst,color='#ffffff',font=ImageFont.truetype(fontpath, fontSize)):
    img=Image.open(oriImageFile)
    DrawImageText(img,poslst,txtlst,color,font)
    t=ConvertImageAsArray(img)
    imgUse.imsave(newImageFile,t)
def DrawImageTextCopyFile(imgFile,srcfile,poslst,txtlst,color='#ffffff',font=ImageFont.truetype(fontpath, fontSize)):
    if(not os.path.exists(imgFile)):
        shutil.copy(srcfile,imgFile)
    DrawImageTextFile(imgFile,imgFile,poslst,txtlst,color,font)
#获取设置像素点的方法
def getPngPix(pngPath = "aa.png",pixelX = 1,pixelY = 1):
    img_src = Image.open(pngPath)
    img_src = img_src.convert('RGBA')
    m=np.array(img_src,dtype=np.int8)
    str_strlist = img_src.load()
    data = str_strlist[pixelX,pixelY]    
    img_src.close()
    return data
@DecorateNumpy
def DrawImageCenter(img,poslst,size=3,color='white',namelst=None,namedelta=2):
    xywhlst=[]
    for p in poslst:
        s=[p[0],p[1],size,size]
        xywhlst.append(s)
    DrawImageCenterBoxes(img,xywhlst,color,namelst,namedelta)
def DrawImageCenterFile(imgfile,svfile,poslst,size=3,color='white',namelst=None,namedelta=2):
    img=Image.open(imgfile)
    DrawImageCenter(img,poslst,size,color,namelst,namedelta)
    t=ConvertImageAsArray(img)
    imgUse.imsave(svfile,t)
def DrawImageCenterCopyFile(imgfile,srcfile,poslst,size=3,color='white',namelst=None,namedelta=2):
    if(not os.path.exists(imgfile)):
        shutil.copy(srcfile,imgfile)
    DrawImageCenterFile(imgfile,imgfile,poslst,size,color,namelst,namedelta)
#draw.line([(0,0),(100,300),(200,500)], fill = '#ffffff', width = 5)
@DecorateNumpy
def DrawImageLine(img,linelst,color='white',width=1):
    idraw=ImageDraw.Draw(img)
    for i in range(1,len(linelst)):
        pts=(linelst[i-1][0],linelst[i-1][1],linelst[i][0],linelst[i][1])
        idraw.line(pts,fill=color,width=width)
    return img
def DrawImageLineFile(imgFile,svFile,linelst,color='white',width=1):
    img=Image.open(imgFile)
    DrawImageLine(img,linelst,color,width)
    t=ConvertImageAsArray(img)
    imgUse.imsave(svFile,t)
def DrawImageLineCopyFile(imgfile,srcfile,linelst,color='white',width=1):
    if(not os.path.exists(imgfile)):
        shutil.copy(srcfile,imgfile)
    DrawImageLineFile(imgfile,imgfile,linelst,color,width)
def MatPlotDrawLines(xlst,ylstMat,title='title',labelAxis=['x','y'],lwlst=None,
                          labeline=None,colorlst=None,lsLst=None):
    plt.title(title)
    plt.xlabel(labelAxis[0])
    plt.ylabel(labelAxis[1])
    for i in range(len(ylstMat)):
        ylst=ylstMat[i]
        lw=2
        color='black'
        ls='-'
        if(lwlst):
            lw=lwlst[i]
        if(colorlst):
            color=colorlst[i]
        if(lsLst):
            ls=lsLst[i]
        if(labeline):
            label=labeline[i]
            plt.plot(xlst,ylst,label=label,c=color,ls=ls,lw=lw)
        else:
            plt.plot(xlst,ylst,c=color,ls=ls,lw=lw)
def MatPlotDrawLinesOneWD(xlst,ylstMat,title='title',labelAxis=['x','y'],lwlst=None,
                          labeline=None,colorlst=None,lsLst=None):
    fig=plt.figure()
    MatPlotDrawLines(xlst,ylstMat,title,labelAxis,lwlst,labeline,colorlst,lsLst)
    return fig
def MatPlotDrawLinesOneWDFile(strfile,shape,title='title',labelAxis=['x','y'],lwlst=None,
                          labeline=None,colorlst=None,lsLst=None):
    m=np.fromfile(strfile,dtype=np.float32)
    m=m.reshape(shape)
    xlst=m[0]
    ylstMat=m[1:-1]
    fig=plt.figure()
    MatPlotDrawLinesOneWD(xlst,ylstMat,title,labelAxis,lwlst,labeline,colorlst,lsLst)
    return fig
#lslst: lineStyle list lslst:宽度list
def MatPlotDrawMultiLines(xlstMat,ylstMat,title='title',labelAxis=['x','y'],lwlst=None,
                          labeline=None,colorlst=None,lsLst=None):
    plt.title(title)
    plt.xlabel(labelAxis[0])
    plt.ylabel(labelAxis[1])
    for i in range(len(ylstMat)):
        ylst=ylstMat[i]
        xlst=xlstMat[i]
        lw=2
        color='black'
        ls='-'
        if(lwlst):
            lw=lwlst[i]
        if(colorlst):
            color=colorlst[i]
        if(lsLst):
            ls=lsLst[i]
        if(labeline):
            label=labeline[i]
            plt.plot(xlst,ylst,label=label,c=color,ls=ls,lw=lw)
        else:
            plt.plot(xlst,ylst,c=color,ls=ls,lw=lw)
#minMaxValue和yMinMax类似
def MatPlotDrawMultiLinesOneWD(xlstMat,ylstMat,title='title',labelAxis=['x','y'],lwlst=None,
                          labeline=None,colorlst=None,lsLst=None,minMaxValue=None):
    fig=plt.figure(figsize=(20,20),dpi=80)
    plt.grid(ls='--')
    if(minMaxValue):
        for i in range(len(ylstMat)):
            for j in range(len(ylstMat[i])):
                ylstMat[i][j]=max(minMaxValue[0],min(minMaxValue[1],ylstMat[i][j]))        
    MatPlotDrawMultiLines(xlstMat,ylstMat,title,labelAxis,lwlst,labeline,colorlst,lsLst)
    return fig
#locator=[epoch,sepDex,]
def SetShowWindow(epoch=10000,sepDex=[40,40],yMinMax=[0,20]):
    x_major_locator=MultipleLocator(epoch/sepDex[0])
    y_major_locator=MultipleLocator((yMinMax[1]-yMinMax[0])/sepDex[1])
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.ylim(yMinMax[0],yMinMax[1])
    plt.xlim(0,epoch)
def CloseFigure(fig):
    plt.close(fig)
def MatPlotShowWindow():
    plt.show()
class MatplotlibUI(object):
    def on_press(event):
        global poslst
        global height
        if( event.button==1):#左键点击
            poslst.append([int(event.xdata), int(event.ydata)])
            if((len(poslst))%5==0):
                if(event.ydata<0.25*height):
                    gvText.append('head')
                elif(event.ydata>0.75*height):
                    gvText.append('tail')
                else:
                    gvText.append('mutual')
                plt.text(int(event.xdata), int(event.ydata),gvText[len(gvText)-1]+str(len(gvText)))
            else:
                plt.plot(int(event.xdata), int(event.ydata), '.y')
            plt.show()
        else:
            pos=poslst.pop()
            if(not pos):return
            if((len(poslst)+1)%5==0):
                plt.plot(int(pos[0]), int(pos[1]), '.r')
                gvText.pop()     
                plt.show()
                return       
            plt.plot(int(pos[0]), int(pos[1]), '.b')
            plt.show()
    def ShowPannel(name,img):
        fig = plt.figure()
        plt.title(name)
        plt.imshow(img, animated= True)
        fig.canvas.mpl_connect('button_press_event', MatplotlibUI.on_press)
        plt.show()
        plt.close(fig)
def DrawPointsList(img,ptslst,sizelst=None,colorlst=None,thickness=3):
    color=(255,255,255)
    size=1
    for i in range(len(ptslst)):
        point=ptslst[i]
        if(sizelst):
            size=sizelst[i]
        if(colorlst):
            color=colorlst[i]        
        cv2.circle(img, point, size, color, thickness)
    return img
def DrawPolylines(img,ptslst,color=None,thickness=3):
    pts = np.array(ptslst, np.int32)
    pts = pts.reshape((-1, 1, 2))
    if(not color):color=(255,255,255)
    cv2.polylines(img, [pts], True, color, thickness)
    return img
def DrawPointlstCopyFile(file,svfile,ptslst,sizelst=None,colorlst=None,thickness=3):
    img=Image.open(file)
    img=img.convert('RGB')
    img=np.array(img,np.uint8)
    DrawPointsList(img,ptslst,sizelst,colorlst,thickness)
    t=ConvertImageAsArray(img)
    imgUse.imsave(svfile,t)
def DrawPointlstToFile(img,svfile,ptslst,sizelst=None,colorlst=None,thickness=3):
    img=np.array(img,np.uint8)
    DrawPointsList(img,ptslst,sizelst,colorlst,thickness)
    t=ConvertImageAsArray(img)
    imgUse.imsave(svfile,t)
def DrawPolylinesCopyFile(file,svfile,ptslst,color=None,thickness=3):
    img=Image.open(file)
    img=img.convert('RGB')
    img=np.array(img,np.uint8)
    DrawPolylines(img,ptslst,color,thickness)
    t=ConvertImageAsArray(img)
    imgUse.imsave(svfile,t)
def DrawPolylinesToFile(img,svfile,ptslst,color=None,thickness=3):
    img=np.array(img,np.uint8)
    DrawPolylines(img,ptslst,color,thickness)
    t=ConvertImageAsArray(img)
    imgUse.imsave(svfile,t)
'''
img1:H,W,C,img2:H,W,C,matches:N,2,points1:M,2,points2:M,2
'''
def DrawMatchPair2Images(img1:np.ndarray,img2:np.ndarray,
                         matches:np.ndarray,points1:np.ndarray,points2:np.ndarray,
                         color=None,thickness:int=1,matchesNum=-1,startNum=0,shuffle=True):
    if shuffle:
        np.random.shuffle(matches)
    imatch1,imatch2=matches[...,0],matches[...,1]    
    if matchesNum>0:
        imatch1,imatch2=imatch1[startNum:startNum+matchesNum],imatch2[startNum:startNum+matchesNum]
    ipts1,ipts2=points1[imatch1],points2[imatch2]
    ipts1,ipts2=ipts1.copy(),ipts2.copy()
    w,h=max(img1.shape[1],img2.shape[1]),max(img1.shape[0],img2.shape[0])
    newimg=np.zeros([2*h,w,3],np.uint8)
    ipts2+=[0,h]
    ipts=np.stack([ipts1,ipts2],1).astype(np.int32)
    newimg[:img1.shape[0],:img1.shape[1]]=img1
    newimg[h:h+img2.shape[0],:img2.shape[1]]=img2
    cv2.polylines(newimg,ipts,isClosed=False,color=color,thickness=thickness)
    return newimg
def DrawMatchPair2ImagesTxt(img1:np.ndarray,img2:np.ndarray,
                         matches:np.ndarray,points1:np.ndarray,points2:np.ndarray,
                         color=None,scale=1.0,fontscale=0.2):
    imatch1,imatch2=matches[...,0],matches[...,1] 
    ipts1,ipts2=points1[imatch1],points2[imatch2]
    ipts1,ipts2=ipts1.copy().astype(np.int32),ipts2.copy().astype(np.int32)
    imgscale1,imgscale2=list(img1.shape[:2])[::-1],list(img2.shape[:2])[::-1]
    imgscale1,imgscale2=np.array(imgscale1,np.float32)*scale,np.array(imgscale2,np.float32)*scale
    imgscale1,imgscale2=imgscale1.astype(np.int32).tolist(),imgscale2.astype(np.int32).tolist()
    nimg1,nimg2=cv2.resize(img1,imgscale1),cv2.resize(img2,imgscale2)
    for i in range(ipts1.shape[0]):
        ipos1,ipos2=ipts1[i],ipts2[i]
        txt=str(i)
        nimg1=cv2.putText(nimg1,txt,ipos1.tolist(),color=color,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=fontscale)
        nimg2=cv2.putText(nimg2,txt,ipos2.tolist(),color=color,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=fontscale)
    return nimg1,nimg2
if(__name__=="__main__"):
    arr=np.zeros([5,4,3],dtype=np.uint8)
    img = Image.fromarray(arr).convert('RGB')
    #plt.figure(figsize=(10,10),dpi=80)
    #plt.grid(ls='--')    
    #plt.text(x,y,'Hello')
    #plt.savefig('D:\\data.jpg')
    #pic= cv2.imread(data['picture'])
    #for j in data['structure']:
    #    cx=XywhToLtRb(j[2])
    #    cx=[int(cx[0]),int(cx[1]),int(cx[2]),int(cx[3])]
    #    cv2.rectangle(pic, (cx[0],cx[1]), (cx[2],cx[3]), (255,255,255), 1)
    #    cv2.putText(pic, j[0], (cx[0]+2, cx[1]+9), font, 0.35, (255,255,255), 1)
    #resfile=os.path.join(newDir,str(count)+'.jpg')
    #cv2.imwrite(resfile,pic)
    #######################################################################3
    #绘制多边形
    #####################################################################3##
    #pts = np.array([[100, 5],  [500, 100], [600, 200], [200, 300]], np.int32)
    # 顶点个数：4，矩阵变成4*1*2维
    #pts = pts.reshape((-1, 1, 2))
    #cv2.polylines(img, [pts], True, (0, 0, 255), 10)
    #########################################################################
    #########################################################################
    #绘制顶点
    #########################################################################
    