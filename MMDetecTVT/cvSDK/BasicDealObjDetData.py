import os,sys
sys.path.append(os.path.dirname(__file__))
import EhualuInterFace as ehl
import BasicUseFunc as basFunc
import numpy as np
import os,sys
import DataIO as dio
import BasicDrawing as basDraw
import re
import matplotlib.image as imgUse
import cv2
#这个文件中所有boxes/box都只有长宽，没有x,y
def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    :param box: tuple or array, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        return 0

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_
def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])
def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    :param boxes: numpy array of shape (r, 4)
    :return: numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)
def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]
    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))
    np.random.seed()
    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]
    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)
        nearest_clusters = np.argmin(distances, axis=1)
        dists=np.linalg.norm(nearest_clusters-last_clusters,axis=None)
        print('dist:',dists,end='\r',flush=True)
        if (last_clusters == nearest_clusters).all():
            break
        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)
        last_clusters = nearest_clusters    
    cluster=clusters
    product=cluster[:,0]*cluster[:,1]
    z=np.concatenate([cluster,np.expand_dims(product,len(product.shape))],-1)
    m=sorted(z,key=lambda x:x[-1])
    m=np.array(m)
    cluster=m[:,:2]
    ShowClusterResult(last_clusters,boxes,k,cluster)
    return cluster
def GetImageStdMean(imgfiles):
    num=len(imgfiles)
    imglst=[]
    count=0
    meanimg=np.zeros([3],np.float32)
    allpix=0.0
    print('mean request:')
    for file in imgfiles:
        img=cv2.imread(file)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=np.array(img,np.float32)/255.0
        sumres=np.array([np.sum(img[:,:,i]) for i in range(3)],np.float32)
        allpix+=float(img.shape[0]*img.shape[1])
        meanimg+=sumres
        basFunc.Process(count,len(imgfiles))
        count+=1
    meanimg/=allpix
    print('\nstd request:')
    stdimg=np.zeros([3],np.float32)
    count=0
    for file in imgfiles:
        img=cv2.imread(file)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=np.array(img,np.float32)/255.0
        img-=meanimg
        img=img**2
        sumres=np.array([np.sum(img[:,:,i]) for i in range(3)],np.float32)
        stdimg+=sumres
        basFunc.Process(count,len(imgfiles))
        count+=1
    stdimg/=allpix
    stdimg=np.sqrt(stdimg)
    return meanimg,stdimg
def GetColor(i):
    color=[[255,0,0],
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
           [128,128,0],
           [0,128,128],
           [128,0,128],
           [64,64,0],
           [0,64,64],
           [64,0,64]]
    return color[i%len(color)]
def ShowClusterResult(clusters,boxes,kNum,cls,height=700,width=700):
    maxw=np.max(boxes[:,0])+1
    maxh=np.max(boxes[:,1])+1
    if(height<0 or width<0):
        height=maxh
        width=maxw
    bmp=np.zeros([height,width,3],np.uint8)
    boxes=np.array(boxes,np.float32)
    boxes[:,0]/=maxw
    boxes[:,1]/=maxh
    boxes[:,0]*=width
    boxes[:,1]*=height
    ccls=np.array(cls,np.float32)
    ccls[:,0]/=maxw
    ccls[:,0]*=width
    ccls[:,1]/=maxh
    ccls[:,1]*=height
    boxes=np.array(boxes,np.int32)
    box=[]
    for i in range(kNum):
        pos=np.where(clusters==i)
        curPos=boxes[pos]
        bmp[curPos[:,1],curPos[:,0]]=np.array(GetColor(i),np.uint8)
        box.append([0,0,ccls[i,0],ccls[i,1]])
    import PIL as pil
    bmp=pil.Image.fromarray(bmp)
    bmp=basDraw.DrawImageRectangles(bmp,box)
    bmp=np.array(bmp,np.uint8)
    bmpfile=os.path.join(os.getcwd(),'ClusterResult.jpg')
    imgUse.imsave(bmpfile,bmp)
    return
#datas必须是二维的
def kmeansWithNum(datas,k,dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = datas.shape[0]
    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))
    np.random.seed()
    # the Forgy method will fail if the whole array contains the same rows
    clusters = datas[np.random.choice(rows, k, replace=False)]
    while True:
        for row in range(rows):
            distances[row] = np.linalg.norm(datas[row]-clusters,axis=1)
        nearest_clusters = np.argmin(distances, axis=1)
        dists=np.linalg.norm(nearest_clusters-last_clusters)
        print('dist:',dists,flush=True,end='\r')
        if (last_clusters == nearest_clusters).all():
            break
        for cluster in range(k):
            clusters[cluster] = dist(datas[nearest_clusters == cluster], axis=0)
        last_clusters = nearest_clusters
    product=np.ones([clusters.shape[0],1],np.float32)
    for i in range(clusters.shape[1]):
        product[:,0]*=clusters[:,i]
    res=np.concatenate([clusters,product],-1)
    #res=res.tolist()
    res=sorted(res,key=lambda x: x[-1])
    res=np.array(res,np.float32)
    clusters=res[:,:-1]
    return clusters
def GetAnchorRatioAndScale(strides,kratio,kscale,boxes):
    scales=[]
    ratios=[]
    areas=[]
    for i in range(len(strides)):
        areas.append(strides[i]*strides[i])
    areas=np.array(areas,np.float32)
    bareas=np.array([b[0]*b[1] for b in boxes],np.float32)
    bareas=bareas.reshape([-1,1])
    delta=np.abs(bareas-areas)
    dvid=np.sqrt(bareas/areas)
    dex=np.argmin(delta,axis=1)
    scales=dvid[np.arange(0,bareas.shape[0]),dex].reshape([-1,1])
    ratios=np.array([b[1]/b[0] for b in boxes],np.float32).reshape([-1,1])
    print('caculate scale:')
    scalecluster=kmeansWithNum(scales,kscale)
    print('\ncaculate ratio:')
    ratiocluster=kmeansWithNum(ratios,kratio)
    return scalecluster,ratiocluster
if __name__ == "__main__":
    curDir=os.getcwd()
    curDir=os.path.abspath(curDir)
    curDir=os.path.dirname(curDir)
    trainDir=os.path.join(curDir,'train')
    files=basFunc.getdatas(trainDir)
    data=[]
    count=0
    rfiles=[]
    filter='.json'
    lenth=512
    for file in files:
        d,name,ftr=basFunc.GetfileDirNamefilter(file)
        txtfile=os.path.join(d,name+'.json')
        if(not os.path.exists(txtfile)):continue
        namelst,box,w,h=ehl.ConvertOWNJson(txtfile)
        img=dio.GetOriginImageData(file)
        boxes=[]
        for m in box:
            wh=[img.shape[1],img.shape[0]]         
            mx=max(wh[0],wh[1])   
            b=np.array([float(m[i]) for i in range(4)],np.float32)
            b[2]-=b[0]
            b[3]-=b[1]
            b/=mx
            b*=lenth
            c=b
            data.append([c[2],c[3]])
        count+=1
        basFunc.Process(count,len(files))
        rfiles.append(file)
        #if(count>=500):break
    data=np.array(data,np.int32)
    ratios,scales=GetAnchorRatioAndScale([8,16,32,64,128],3,3,data)
    cluster=kmeans(data,3)     
    meanimg,stdimg=GetImageStdMean(rfiles)   
    print('\ncluster:\n',cluster)
    print('meanimg:',meanimg,'stdimg:',stdimg)
    print('meanimgMM:',meanimg*255.0,'stdimgMM:',stdimg*255.0)
    print('ratios:',ratios,'\nscales:',scales)
        
