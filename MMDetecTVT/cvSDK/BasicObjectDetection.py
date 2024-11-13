import BasicUseFunc as basFunc
import numpy as np
import random
import os,sys 
import BasicDrawing as basDraw
import BasicAlgorismGeo as basGeo
from enum import Enum
import itertools
import time
#####################################################################
########################BOX类型XYWH开始###############################
#####################################################################
class BASICBOX(object):
    def __init__(sf,*args):
        if(len(args)!=1):
            sf.cls,sf.prob,sf.box=args[0],args[1],args[2]
        if(len(args)==1):
            [sf.cls,sf.prob,sf.box]=args[0]
    def __getitem(sf,dex):
        return sf.box[dex]
    def GetBasicBoxArray(basBoxlst,convertNumpy=True):
        allbox=[]
        for b in basBoxlst:
            allbox.append([b.cls,b.prob,b.box])
        if(convertNumpy):
            allbox=np.array(allbox,np.float32)
        return allbox
    def FromBasicBoxArray(boxArr,bProb=True,bNumpyInput=False):
        if(bNumpyInput):
            boxArr=np.array(boxArr,np.float32)
            length=boxArr.shape[0]
        else:
            length=len(boxArr)
        boxes=[]
        for i in range(length):
            b=boxArr[i]
            if(not bProb):
                bx=BASICBOX([b[0],1.0,[b[1],b[2],b[3],b[4]]])
            else:
                bx=BASICBOX([b[0],b[1],[b[2][0],b[2][1],b[2][2],b[2][3]]])
            boxes.append(bx)
        return boxes
    def RandomGenerateBox(strides,images,boxMaxNum,minpix=10):
        anchor=AnchorRatio(strides,[[1.0,1.0]],1.0)
        anchors=anchor.GenerateAnchors(images)
        anchors=anchors[0]
        boxes=[]
        for img in images:
            boxNum=random.randint(5,boxMaxNum)
            flag=np.zeros([anchors.shape[0]])
            _,ow,oh=img.shape
            b=[]
            for i in range(boxNum):
                idex=random.randint(0,anchors.shape[0])
                if(flag[idex]!=0):
                    i-=1
                    continue
                flag[idex]=1
                m=anchors[idex].tolist()
                x,y,w,h=basGeo.LtRbToxywh(m)
                x=random.uniform(x-w/2,x+w/2)
                y=random.uniform(y-h/2,y+h/2)
                x=max(minpix,min(x,ow-minpix))
                y=max(minpix,min(y,oh-minpix))
                w=random.uniform(minpix,min(x,ow-x))
                h=random.uniform(minpix,min(y,oh-y))
                w=2*w-1
                h=2*h-1
                box=BASICBOX(0,1,[x,y,w,h])
                b.append(box)
            boxes.append(b)
        return boxes
                
#boxlst必须输入BASICBOX类型
#返回的是BASICBOX
def BasicBOXNMS(boxlst,probThr=0.5,nms=.45):
    boxes=[]
    boxlst=sorted(boxlst,key=lambda x: x.prob,reverse=True)
    flag=np.zeros([len(boxlst)],np.uint8)
    for i in range(len(boxlst)):
        curbox=boxlst[i]
        if(curbox.prob<probThr):
            flag[i]=1
            continue
    for i in range(len(boxlst)):
        if(flag[i]!=0):continue
        curbox=boxlst[i]
        for j in range(i+1,len(boxlst)):
            sbox=boxlst[j]
            if(curbox.cls!=sbox.cls):continue
            iou=basGeo.iouheightwidth(curbox.box,sbox.box)
            if(iou>nms):
                flag[j]=1
    for i in range(len(boxlst)):
        if(flag[i]!=0):continue
        boxes.append(boxlst[i])
    return boxes
def DeCodeBox(xdex,ydex,box,sepx,sepy):
    cx=float(xdex)*sepx
    cy=float(ydex)*sepy
    tbox=np.zeros([4],dtype=np.float32)
    tbox[0]=box[0]*sepx+cx
    tbox[1]=box[1]*sepy+cy
    tbox[2]=np.exp(box[2])*sepx
    tbox[3]=np.exp(box[3])*sepy
    return tbox
def EnCodeBox(xdex,ydex,box,sepx,sepy):
    cx=float(xdex)*sepx
    cy=float(ydex)*sepy
    tbox=np.zeros([4],dtype=np.float32)
    tbox[0]=(box[0]-cx)/sepx
    tbox[1]=(box[1]-cy)/sepy
    tbox[2]=np.log(box[2]/sepx)
    tbox[3]=np.log(box[3]/sepy)
    return tbox
mxRate=0
def GetShowCurve(validDex,mapRate,ShowCurveLoop=1):
    outDex=[]
    mapShowlst=[]
    global mxRate
    for i in range(len(validDex)):
        flag=False
        if(mapRate[i]>mxRate):
            flag=True
        if(flag or i%ShowCurveLoop==0):
            outDex.append(validDex[i])
            mapShowlst.append(mapRate[i])
    return outDex,mapShowlst
def DrawMapPic(validDex,maplist,mapDir,epoches,ShowCurveLoop=1,TimeLeft=-1):
    if(len(validDex)==0 or len(vprRate)==0):return
    [mappic]=basFunc.GetCurDirNames(['mapShow.jpg'])
    names=[]
    nvalidDex,mapRates=GetShowCurve(validDex,maplist,ShowCurveLoop)
    color=['green']
    fig=basDraw.MatPlotDrawMultiLinesOneWD([nvalidDex],[mapRates],'MapChart,TimeLeft:'+str(round(TimeLeft,4))+' hours',
    ['iter','Err'],colorlst=color,labeline=['Map'],minMaxValue=[0,1])
    basDraw.SetShowWindow(epoches,sepDex=[20,20],yMinMax=[0,1])
    plt.savefig(aprpic)
    basDraw.CloseFigure(fig)
#####################################################################
########################BOX类型XYWH结束###############################
#####################################################################
#-------------------------------------------------------------------#
#####################################################################
########################BOX类型LTRB开始###############################
#####################################################################
class AnchorRatio(object):
    #strides=[8,16,32,64],ratio=[[1.0,1.0],[1.4,0.7],[0.7,1.4]]
    def __init__(sf,strides,ratios,basScale=4.0):
        sf.strides=strides
        sf.ratios=ratios
        sf.basScale=basScale
    #images=NCHW仅支持Numpy还有Pytorch的tensor
    def GetIterDatas(sf):
        return itertools.product(sf.strides,sf.ratios)
    def ArrangeAnchor(sf,iterdata,height,width):
        h=height
        w=width
        stride,ratio=iterdata
        cstride=stride/2
        xs=np.arange(cstride,w,stride)
        ys=np.arange(cstride,h,stride)
        xv,yv=np.meshgrid(xs,ys)
        xv=np.reshape(xv,[1,-1])
        yv=np.reshape(yv,[1,-1])
        xstride=sf.basScale*cstride*ratio[0]
        ystride=sf.basScale*cstride*ratio[1]
        an=np.concatenate((xv-xstride,yv-ystride,xv+xstride,yv+ystride),0)
        anchor=np.transpose(an,[1,0])        
        return anchor

    def GenerateAnchors(sf,images):
        h=images.shape[2]
        w=images.shape[3]
        anchors=[]
        for i in range(images.shape[0]):
            ancs=[]
            for iterdata in sf.GetIterDatas():
                anchor=sf.ArrangeAnchor(iterdata,h,w)     
                ancs.append(anchor)
            anchs=np.concatenate(ancs,0)
            anchos=np.reshape(anchs,[1,anchs.shape[0],anchs.shape[1]])
            anchors.append(anchos)
        anchors=np.concatenate(anchors,0)
        return anchors
class AnchorRatioScale(AnchorRatio):
    def __init__(sf,strides,ratios,scales,basScale=4.0):
        sf.scales=scales
        super().__init__(strides,ratios,basScale)
    def GetIterDatas(sf):
        return itertools.product(sf.strides,sf.ratios,sf.scales)
    def ArrangeAnchor(sf,iterdata,height,width):
        h=height
        w=width
        stride,ratio,scale=iterdata
        cstride=stride/2
        xs=np.arange(cstride,w,stride)
        ys=np.arange(cstride,h,stride)
        xv,yv=np.meshgrid(xs,ys)
        xv=np.reshape(xv,[1,-1])
        yv=np.reshape(yv,[1,-1])
        xstride=sf.basScale*cstride*ratio[0]*scale
        ystride=sf.basScale*cstride*ratio[1]*scale
        an=np.concatenate((xv-xstride,yv-ystride,xv+xstride,yv+ystride),0)
        anchor=np.transpose(an,[1,0])        
        return anchor

#####################################################################
########################BOX类型LTRB结束###############################
#####################################################################
class MAParam(object):
    def __init__(sf):
        sf.AP50=0
        sf.AP75=0
        sf.APSmall=0
        sf.APMeduim=0
        sf.APLarge=0
        sf.MAP=0
        sf.data=[]
        sf.fNDct={}
    #pred=[classname,prob,x,y,w,h]
    #gt=[classname,x,y,w,h]
    def __call__(sf,pred,gt):
        gtlst=np.zeros([len(pred)],np.int32)
        predlst=np.zeros([len(pred)],np.float32)
        for t in range(len(gt)):
            maxiou=-1
            maxipredx=-1
            maxgtdx=-1
            for i in range(len(pred)):   
                if(predlst[i]>0):continue
                box=[pred[i][k] for k in range(2,6)]
                gtbox=[gt[t][k] for k in range(1,5)]
                if(gt[t][0]!=pred[i][0]):continue
                iou=basGeo.iouheightwidth(box,gtbox)
                if(maxiou<0 or iou>maxiou):
                    maxiou=iou
                    maxipredx=i
            if(maxiou>0):
                predlst[maxipredx]=maxiou
                gtlst[maxipredx]=t+1
            else:
                if(gt[t][0] not in sf.fNDct):
                    sf.fNDct[gt[t][0]]=[[gt[t][3],gt[t][4]]]
                else:
                    sf.fNDct[gt[t][0]].append([gt[t][3],gt[t][4]])
        for i in range(len(pred)):
            m=pred[i].copy()
            m.append(float(predlst[i]))
            if(gtlst[i]>0):
                m.append(gt[gtlst[i]-1][2])
                m.append(gt[gtlst[i]-1][3])
            else:
                m.append(-1)
                m.append(-1)
            sf.data.append(m)
    def SortData(sf,statData,iouThr,othFn):
        apres=0
        for i in range(len(statData)):
            statData[i]=sorted(statData[i],key=lambda x:x[0],reverse=True)
            statData[i]=np.array(statData[i])
            m=statData[i]
            result=np.zeros([m.shape[0],2],np.float32)
            for j in range(m.shape[0]):
                maska=m[:,0]>=m[j,0]
                maskb=m[:,5]>=iouThr
                maskc=np.logical_and(maska,maskb)
                tp=np.sum(maskc,0)
                maska=m[:,0]>=m[j,0]
                maskb=m[:,5]<iouThr
                maskc=np.logical_and(maska,maskb)
                fp=np.sum(maskc,0)
                maska=m[:,0]<m[j,0]
                maskb=m[:,5]>=iouThr
                maskc=np.logical_and(maska,maskb)
                fn=np.sum(maskc,0)+othFn[i]
                if(tp+fp==0):prec=0
                else:prec=tp/(tp+fp)
                result[j,0]=round(prec,2)
                if(tp+fn==0):recall=0
                else:recall=tp/(tp+fn)
                result[j,1]=round(recall,2)
            prlst=[]
            flag=False
            result=result.tolist()
            result=sorted(result,key=lambda x: x[1])
            for j in range(len(result)):
                if(len(prlst)==0):
                    prlst.append([result[j][0],result[j][1]])
                else:
                    cutDex=-1
                    for k in range(len(prlst)):
                        if(prlst[k][0]<=result[j][0]):
                            cutDex=k+1
                            prlst[k]=result[j]
                            break
                        if(prlst[k][0]>result[j][0] and prlst[k][1]==result[j][1]):
                            cutDex=k+1
                            break
                    if(cutDex<=0):
                        prlst.append(result[j])
                    else:
                        prlst=np.array(prlst,np.float32)[0:cutDex].tolist()
            apclass=0
            for j in range(len(prlst)):
                if(j==0):                    
                    apclass+=prlst[j][0]*(prlst[j][1]-0)
                else:
                    apclass+=prlst[j][0]*(prlst[j][1]-prlst[j-1][1])
            apres+=apclass
        if(len(statData)!=0):
            apres/=float(len(statData))
        else:
            apres=0
        return apres
    def StaticPrecisonDetail(sf,areaMin,areaMax,iouThr):
        classes=[]
        ndata=[]
        fndct={}
        for kk,gt in sf.fNDct.items():
            gtlst=[]
            for g in gt:
                curarea=g[0]*g[1]
                if(not ((not areaMin or curarea>areaMin )and (not areaMax or curarea<areaMax))):
                    continue
                gtlst.append([g[0],g[1]])
            if(len(gtlst)==0):continue
            fndct[kk]=len(gtlst)
        for d in sf.data:
            curarea=d[4]*d[5]
            if(not ((not areaMin or curarea>areaMin )and (not areaMax or curarea<areaMax))):
                curarea=d[7]*d[8]
                if(d[7]<0):continue
                if((not areaMin or curarea>areaMin )and (not areaMax or curarea<areaMax)):
                    if(d[0] not in fndct):
                        fndct[d[0]]=1
                    else:
                        fndct[d[0]]+=1
                continue
            ndata.append(d)
        for d in ndata:
            if(d[0] not in classes):
                classes.append(d[0])
        fnOth=[0 for i in range(len(classes))]
        for kk,vv in fndct.items():
            if(kk not in classes):
                classes.append(kk)
                fnOth.append(0)
            fnOth[classes.index(kk)]=vv
        mdata=[[] for i in range(len(classes))]
        print('\n---------------------------------------------------------\nDataNumber:',len(ndata),'fndct:',fnOth)
        for i in range(len(ndata)):
            nsdata=[ndata[i][j] for j in range(1,9)]
            mdata[classes.index(ndata[i][0])].append(nsdata)
        sdata=[]
        for i in range(len(mdata)):
            sdata.append(np.array(mdata[i],np.float32))
        apres=sf.SortData(sdata,iouThr,fnOth)
        return apres
    def StaticPrecison(sf):
        sf.MAP=0
        count=0
        for iou in np.arange(0.5,1.0,0.05):
            curMap=sf.StaticPrecisonDetail(None,None,iou)
            print('MAP:',round(iou,2),' res:',curMap)
            if(round(iou,2)==0.5):
                sf.AP50=curMap
            if(round(iou,2)==0.75):
                sf.AP75=curMap
            sf.MAP+=curMap
            count+=1
        sf.MAP/=float(count)
        print('MAP:',sf.MAP,end=' ')
        sf.APLarge=sf.StaticPrecisonDetail(96*96,None,0.5)
        print('APLarge:',sf.APLarge,end=' ')
        sf.APMeduim=sf.StaticPrecisonDetail(32*32,96*96,0.5)
        print('APMeduim:',sf.APMeduim,end=' ')
        sf.APSmall=sf.StaticPrecisonDetail(None,32*32,0.5)
        print('APSmall:',sf.APSmall)
def BOXIoUEx(boxlstA,boxlstB):
    areaA=(boxlstA[...,2]-boxlstA[...,0])*(boxlstA[...,3]-boxlstA[...,1])
    areaB=(boxlstB[...,2]-boxlstB[...,0])*(boxlstB[...,3]-boxlstB[...,1])
    cboxlstA=boxlstA[:,None,...]
    cboxlstB=boxlstB[None,:,...]
    careaA=areaA[:,None,...]
    careaB=areaB[None,:,...]
    lt=np.maximum(cboxlstA[...,:2],cboxlstB[...,:2])
    rb=np.minimum(cboxlstA[...,2:],cboxlstB[...,2:])
    delta=rb-lt
    delta=np.where(delta<0,0,delta)
    inter=delta[...,0]*delta[...,1]
    area=careaA+careaB
    iou=inter/(area-inter)
    return iou
def BOXIoU(boxlstA,boxlstB):
    iou=BOXIoUEx(boxlstA,boxlstB)
    iou=np.max(iou,1)
    return iou
#boxlst=[class,scores,boxes]
def GeneralNMSBox(boxlst,threshold=0.5,iouthreshold=0.45,uniqueArea=True):
    npboxdct={} 
    classes=[]
    for b in boxlst:
        if(b[0] not in classes):
            classes.append(b[0])
    for b in boxlst:
        nbox=list(b[2])
        clss=b[0]
        if(uniqueArea):clss='unique'
        if(clss not in npboxdct.keys()):
            npboxdct[clss]=[np.array([b[1]]+nbox+[classes.index(b[0])],np.float32)]
        else:
            npboxdct[clss].append(np.array([b[1]]+nbox+[classes.index(b[0])],np.float32))    
    ioudct={}
    for kk in npboxdct.keys():        
        npboxdct[kk]=sorted(npboxdct[kk],key=lambda x:x[0],reverse=True)
        npboxdct[kk]=np.array(npboxdct[kk],np.float32)
        dex=np.where(npboxdct[kk][:,0]>threshold)
        dex=dex[0]
        if dex.shape[0]==0:
            npboxdct[kk]=np.zeros([0,6],np.float32)
            continue   
        npboxdct[kk]=npboxdct[kk][dex,:]
        if npboxdct[kk].shape[0]==1:
            continue
        npbox=npboxdct[kk][:,1:5]
        iou=BOXIoUEx(npbox,npbox)
        iou=np.tril(iou,-1)
        iou=np.max(iou,1)
        dex=np.where(iou<iouthreshold)
        dex=dex[0]
        npboxdct[kk]=npboxdct[kk][dex,:]
    result=[]
    for kk,vv in npboxdct.items():
        for v in vv:
            result.append([classes[int(v[-1])],v[0],list(v[1:5])])    
    return result      
def GetGeneralBoxToCSB(genbox):
    boxes=[]
    classes=[]
    scores=[]
    for i in range(len(genbox)):
        classes.append(genbox[i][0])
        scores.append(genbox[i][1])
        boxes.append(genbox[i][2])
    return classes,boxes,scores
def GetCStoTxt(classes,scores,rd=2):
    assert(len(classes)==len(scores))
    strTxt=[]
    for i in range(len(classes)):
        strTxt.append(str(classes[i])+':'+str(round(scores[i],rd)))
    return strTxt
'''
res=GeneralNMSBox(boxlst)
c,b,s=GetGeneralBoxToCSB(res)
txts=GetCStoTxt(c,s)
'''        
def GetBDist(lt,rb):
    delta=lt[None,...]-rb[...,None,:]
    dist=np.linalg.norm(delta,axis=-1)
    return dist
def GetBoxDist(ltrbx,ltrby,defaultMax=100000):
    iou=BOXIoUEx(ltrbx,ltrby)
    msk=iou>0
    ltltdst=GetBDist(ltrbx[...,[0,1]],ltrby[...,[0,1]])    
    ltrbdst=GetBDist(ltrbx[...,[0,1]],ltrby[...,[2,3]])
    rbrbdst=GetBDist(ltrbx[...,[2,3]],ltrby[...,[2,3]])
    dsts=np.stack([ltltdst,ltrbdst,rbrbdst],-1).min(-1)  
    dsts[msk]=0
    msk=np.ones_like(dsts)
    msk=np.triu(msk,k=1)
    dsts[msk==0]=defaultMax
    return dsts
def CheckInsideBoxes(boxlstA,boxlstB):
    cboxlstA=boxlstA[:,None,...]
    cboxlstB=boxlstB[None,:,...]    
    lt=np.maximum(cboxlstA[...,:2],cboxlstB[...,:2])
    rb=np.minimum(cboxlstA[...,2:],cboxlstB[...,2:])
    lt=lt.clip(cboxlstB[...,[0,1]],cboxlstB[...,[2,3]])
    rb=rb.clip(cboxlstB[...,[0,1]],cboxlstB[...,[2,3]])
    areaA=(boxlstA[...,2]-boxlstA[...,0])*(boxlstA[...,3]-boxlstA[...,1])
    areaA=areaA[...,None]
    delta=rb-lt
    areaB=delta[...,0]*delta[...,1]
    ratio=areaB/areaA
    return ratio
            

        

    
            


        


if __name__ == "__main__":
    #anchor=AnchorRatio([8,16,32],[[1.0,1.0],[1.4,0.7],[0.7,1.4]])
    imgs=np.random.randn(4,3,128,128)
    boxes=BASICBOX.RandomGenerateBox([8,16],imgs,10)
    anchor=AnchorRatioScale([8,16,32],[[1.0,1.0],[1.4,0.7],[0.7,1.4]],[3.0,2.0])
    
    anchor.GenerateAnchors(imgs)
