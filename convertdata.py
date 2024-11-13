import os,sys
import cvSDK.BasicUseFunc as basFunc
import cvSDK.DataIO as dio
import cvSDK.BasicPicDeal as basPic
from sixdrepnet.utils import *
from sixdrepnet.enhaunce import *
import cvSDK.BasicDrawing as basDraw
import shutil
import cv2
def GetBox(file,pt2d):
    img=cv2.imread(file)
    msk=np.logical_or(pt2d<0,pt2d>img.shape[1])
    msk=~msk
    xmin=pt2d[0][msk[0]].min()
    xmax=pt2d[0][msk[0]].max()
    ymin=pt2d[1][msk[1]].min()
    ymax=pt2d[1][msk[1]].max()
    return xmin,ymin,xmax,ymax
#所有格式是弧度
def WLP():
    outDir=basFunc.MakeEmptyDir("wlpDir")
    debugDir=basFunc.MakeEmptyDir("debugDir")
    cnt=0
    for file,d,name,ftr in basFunc.GetFileDirLst("300W_LP",False,True):
        shutil.copy(file,os.path.join(outDir,str(cnt)+".jpg"))
        matfile=os.path.join(d,name+".mat")
        pt2d=get_pt2d_from_mat(matfile)
        x_min = min(pt2d[0,:])
        y_min = min(pt2d[1,:])
        x_max = max(pt2d[0,:])
        y_max = max(pt2d[1,:])
        box=[x_min,y_min,x_max,y_max]
        pose = get_ypr_from_mat(matfile)
        pitch = pose[0]# * 180 / np.pi
        yaw = pose[1] #* 180 / np.pi
        roll = pose[2]# * 180 / np.pi
        img=cv2.imread(file)
        mat=get_R(-pitch,yaw,-roll)
        tp=draw_axis(img,yaw*180/np.pi,pitch*180/np.pi,roll*180/np.pi,0,0,1)
        label=[pitch,yaw,roll]
        pklfile=os.path.join(outDir,str(cnt)+".pkl")
        dio.SaveVariableToPKL(pklfile,[box,label])
        debugfile=os.path.join(debugDir,str(cnt)+".jpg")
        img=basDraw.DrawImageRectangles(img,[box],useNpy=True)
        img=draw_axis(img,yaw*180/np.pi,pitch*180/np.pi,roll*180/np.pi,None,None)
        cv2.imwrite(debugfile,img)
        cnt+=1
    return
#所有格式是弧度
def WLP2():
    outDir=basFunc.MakeEmptyDir("wlpDir")
    debugDir=basFunc.MakeEmptyDir("debugDir")
    cnt=0
    for file,d,name,ftr in basFunc.GetFileDirLst("300W_LP",False,True):
        shutil.copy(file,os.path.join(outDir,str(cnt)+".jpg"))
        matfile=os.path.join(d,name+".mat")
        pt2d=get_pt2d_from_mat(matfile)
        x_min = min(pt2d[0,:])
        y_min = min(pt2d[1,:])
        x_max = max(pt2d[0,:])
        y_max = max(pt2d[1,:])
        box=[x_min,y_min,x_max,y_max]
        pose = get_ypr_from_mat(matfile)
        pitch = pose[0]# * 180 / np.pi
        yaw = pose[1] #* 180 / np.pi
        roll = pose[2]# * 180 / np.pi
        debugfile=os.path.join(debugDir,str(cnt)+"_x.jpg")
        oimg=cv2.imread(file)
        oimg=draw_axis(oimg,yaw*180/np.pi,pitch*180/np.pi,roll*180/np.pi)
        cv2.imwrite(debugfile,oimg)
        rot=get_R(-pitch,yaw,-roll)
        rott=np.transpose(rot,[1,0])
        img=cv2.imread(file)
        img,ngt,angle=RotateCV(img,rott)
        ngtt=np.transpose(ngt,[1,0])
        outelar=compute_euler_angles_from_rotation_matricesnpy(ngtt[None,...])[0]
        npitch,nyaw,nroll=outelar.tolist()
        npitch=-npitch
        nroll=-nroll
        debugfile=os.path.join(debugDir,str(cnt)+".jpg")        
        img=draw_axis(img,nyaw*180/np.pi,npitch*180/np.pi,nroll*180/np.pi,None,None)
        #img=draw_axis(img,yaw*180/np.pi,pitch*180/np.pi,roll*180/np.pi,None,None)
        cv2.imwrite(debugfile,img)
        cnt+=1
    return
def BIWI():
    outDir=basFunc.MakeEmptyDir("biwi")
    debugDir=basFunc.MakeEmptyDir("debugDir")
    cnt=0
    for file,d,name,ftr in basFunc.GetFileDirLst("faces_0",False,True,"*.png"):
        img=cv2.imread(file)
        outimg=os.path.join(outDir,str(cnt)+"_b.jpg")
        fname=name
        dex=-1
        dex=name.find("_rgb")
        if dex>=0:
            name=name[:dex]
        basd=os.path.basename(d)
        mskdir='head_pose_masks'
        mskfile=os.path.join(mskdir,basd,name+"_depth_mask.png")
        if(not os.path.exists(mskfile)):
            continue
        msk=cv2.imread(mskfile,cv2.IMREAD_UNCHANGED)
        msk=msk==255
        ydexs,xdexs=np.where(msk)
        box=[xdexs.min(),ydexs.min(),xdexs.max(),ydexs.max()]
        posetxt=os.path.join(d,name+"_pose.txt")
        with open(posetxt) as f:
            lines=f.readlines()
        lines=lines[:3]
        vals=[]
        for i in range(len(lines)):
            lines[i]=lines[i].strip()
            for l in lines[i].split(" "):
                vals.append(float(l))
        assert(len(vals)==9)
        mat=np.array(vals).reshape(3,3)
        debugfile=os.path.join(debugDir,str(cnt)+'.jpg')
        outelar=compute_euler_angles_from_rotation_matricesnpy(mat[None,...])[0]
        npitch,nyaw,nroll=outelar.tolist()
        npitch,nyaw,nroll=-npitch,nyaw,-nroll
        label=[npitch,nyaw,nroll]
        cv2.imwrite(outimg,img)
        dio.SaveVariableToPKL(os.path.join(outDir,str(cnt)+"_b.pkl"),[box,label])
        img=basDraw.DrawImageRectangles(img,[box],useNpy=True)
        img=draw_axis(img,nyaw*180/np.pi,npitch*180/np.pi,nroll*180/np.pi,None,None)
        cv2.imwrite(debugfile,img)
        cnt+=1

def AFLW():
    outDir=basFunc.MakeEmptyDir("Aflw")
    debugDir=basFunc.MakeEmptyDir("debugDir")
    cnt=0
    for file,d,name,ftr in basFunc.GetFileDirLst("AFLW2000",False,True):
        shutil.copy(file,os.path.join(outDir,str(cnt)+"_aflw.jpg"))
        matfile=os.path.join(d,name+".mat")
        pt2d=get_pt2d_from_mat(matfile)
        x_min,y_min,x_max,y_max=GetBox(file,pt2d)
        cx,cy,w,h=(x_min+x_max)/2,(y_min+y_max)/2,x_max-x_min,y_max-y_min
        w*=2
        h*=2
        x_min,y_min,x_max,y_max=cx-w/2,cy-h/2,cx+w/2,cy+h/2
        box=[x_min,y_min,x_max,y_max]
        pose = get_ypr_from_mat(matfile)
        pitch = pose[0]# * 180 / np.pi
        yaw = pose[1] #* 180 / np.pi
        roll = pose[2]# * 180 / np.pi
        img=cv2.imread(file)
        x_min,y_min=max(x_min,0),max(y_min,0)
        x_max,y_max=min(x_max,img.shape[1]-1),min(y_max,img.shape[0]-1)
        img=img[int(y_min):int(y_max),int(x_min):int(x_max)]
        mat=get_R(-pitch,yaw,-roll)
        tp=draw_axis(img,yaw*180/np.pi,pitch*180/np.pi,roll*180/np.pi,0,0,1)
        label=[pitch,yaw,roll]
        pklfile=os.path.join(outDir,str(cnt)+"_aflw.pkl")
        dio.SaveVariableToPKL(pklfile,[[],label])
        debugfile=os.path.join(debugDir,str(cnt)+".jpg")
        img=draw_axis(img,yaw*180/np.pi,pitch*180/np.pi,roll*180/np.pi,None,None)
        cv2.imwrite(debugfile,img)
        cnt+=1
    return
def AFLWEnhaunce():
    outDir=basFunc.MakeEmptyDir("val")
    debugDir=basFunc.MakeEmptyDir("debugDir")
    cnt=0
    allfiles=[]
    for file,d,name,ftr in basFunc.GetFileDirLst("AFLW2000",False,True):
        allfiles.append([file,d,name,ftr])
    allcnts=len(allfiles)*2
    for i in range(int(allcnts)):
        basFunc.Process(i,int(allcnts))
        file,d,name,ftr=allfiles[i%len(allfiles)]
        outfile=os.path.join(outDir,str(cnt)+"_enh.jpg")
        matfile=os.path.join(d,name+".mat")
        pt2d=get_pt2d_from_mat(matfile)
        x_min,y_min,x_max,y_max=GetBox(file,pt2d)
        cx,cy,w,h=(x_min+x_max)/2,(y_min+y_max)/2,x_max-x_min,y_max-y_min
        w*=2
        h*=2
        x_min,y_min,x_max,y_max=cx-w/2,cy-h/2,cx+w/2,cy+h/2
        x_min,y_min=max(x_min,0),max(y_min,0)
        img=cv2.imread(file)
        x_max,y_max=min(x_max,img.shape[1]-1),min(y_max,img.shape[0]-1)
        box=[x_min,y_min,x_max,y_max]
        pose = get_ypr_from_mat(matfile)
        pitch = pose[0]# * 180 / np.pi
        yaw = pose[1] #* 180 / np.pi
        roll = pose[2]# * 180 / np.pi        
        img=img[int(y_min):int(y_max),int(x_min):int(x_max)]
        img=EnhanunceCV(img)
        cv2.imwrite(outfile,img)
        mat=get_R(-pitch,yaw,-roll)
        tp=draw_axis(img,yaw*180/np.pi,pitch*180/np.pi,roll*180/np.pi,0,0,1)
        label=[pitch,yaw,roll]
        pklfile=os.path.join(outDir,str(cnt)+"_enh.pkl")
        dio.SaveVariableToPKL(pklfile,[[],label])
        debugfile=os.path.join(debugDir,str(cnt)+".jpg")
        img=draw_axis(img,yaw*180/np.pi,pitch*180/np.pi,roll*180/np.pi,None,None)
        cv2.imwrite(debugfile,img)
        cnt+=1
    return
def AFLWEnhaunceGray():
    [outDir]=basFunc.GetCurDirNames(["val"])
    debugDir=basFunc.MakeEmptyDir("debugDir")
    cnt=0
    allfiles=[]
    for file,d,name,ftr in basFunc.GetFileDirLst("AFLW2000",False,True):
        allfiles.append([file,d,name,ftr])
    allcnts=len(allfiles)*1
    for i in range(int(allcnts)):
        basFunc.Process(i,int(allcnts))
        file,d,name,ftr=allfiles[i%len(allfiles)]
        outfile=os.path.join(outDir,str(cnt)+"_gray.jpg")
        matfile=os.path.join(d,name+".mat")
        pt2d=get_pt2d_from_mat(matfile)
        x_min,y_min,x_max,y_max=GetBox(file,pt2d)
        cx,cy,w,h=(x_min+x_max)/2,(y_min+y_max)/2,x_max-x_min,y_max-y_min
        w*=2
        h*=2
        x_min,y_min,x_max,y_max=cx-w/2,cy-h/2,cx+w/2,cy+h/2
        x_min,y_min=max(x_min,0),max(y_min,0)
        img=cv2.imread(file)
        x_max,y_max=min(x_max,img.shape[1]-1),min(y_max,img.shape[0]-1)
        box=[x_min,y_min,x_max,y_max]
        pose = get_ypr_from_mat(matfile)
        pitch = pose[0]# * 180 / np.pi
        yaw = pose[1] #* 180 / np.pi
        roll = pose[2]# * 180 / np.pi        
        img=img[int(y_min):int(y_max),int(x_min):int(x_max)]
        img=EnhanunceCV(img)
        img=Gray(img,1.0)
        cv2.imwrite(outfile,img)
        mat=get_R(-pitch,yaw,-roll)
        tp=draw_axis(img,yaw*180/np.pi,pitch*180/np.pi,roll*180/np.pi,0,0,1)
        label=[pitch,yaw,roll]
        pklfile=os.path.join(outDir,str(cnt)+"_gray.pkl")
        dio.SaveVariableToPKL(pklfile,[[],label])
        debugfile=os.path.join(debugDir,str(cnt)+".jpg")
        img=draw_axis(img,yaw*180/np.pi,pitch*180/np.pi,roll*180/np.pi,None,None)
        cv2.imwrite(debugfile,img)
        cnt+=1
    return
def AFLWEnhaunceRotate():
    [outDir]=basFunc.GetCurDirNames(["val"])
    debugDir=basFunc.MakeEmptyDir("debugDir")
    cnt=0
    allfiles=[]
    for file,d,name,ftr in basFunc.GetFileDirLst("AFLW2000",False,True):
        allfiles.append([file,d,name,ftr])
    allcnts=len(allfiles)*2
    for i in range(int(allcnts)):
        basFunc.Process(i,int(allcnts))
        file,d,name,ftr=allfiles[i%len(allfiles)]
        outfile=os.path.join(outDir,str(cnt)+"_rot.jpg")
        matfile=os.path.join(d,name+".mat")
        pt2d=get_pt2d_from_mat(matfile)
        x_min,y_min,x_max,y_max=GetBox(file,pt2d)
        cx,cy,w,h=(x_min+x_max)/2,(y_min+y_max)/2,x_max-x_min,y_max-y_min
        w*=2
        h*=2
        x_min,y_min,x_max,y_max=cx-w/2,cy-h/2,cx+w/2,cy+h/2
        x_min,y_min=max(x_min,0),max(y_min,0)
        img=cv2.imread(file)
        x_max,y_max=min(x_max,img.shape[1]-1),min(y_max,img.shape[0]-1)
        box=[x_min,y_min,x_max,y_max]
        pose = get_ypr_from_mat(matfile)
        pitch = pose[0]# * 180 / np.pi
        yaw = pose[1] #* 180 / np.pi
        roll = pose[2]# * 180 / np.pi        
        img=img[int(y_min):int(y_max),int(x_min):int(x_max)]
        img=EnhanunceCV(img)
        img=Gray(img)
        rot=get_R(-pitch,yaw,-roll)
        rott=np.transpose(rot,[1,0])
        img,ngt,angle=RotateCV(img,rott)
        ngtt=np.transpose(ngt,[1,0])
        outelar=compute_euler_angles_from_rotation_matricesnpy(ngtt[None,...])[0]
        npitch,nyaw,nroll=outelar.tolist()
        npitch=-npitch
        nroll=-nroll
        cv2.imwrite(outfile,img)        
        label=[npitch,nyaw,nroll]
        pklfile=os.path.join(outDir,str(cnt)+"_rot.pkl")
        dio.SaveVariableToPKL(pklfile,[[],label])
        debugfile=os.path.join(debugDir,str(cnt)+".jpg")
        img=draw_axis(img,nyaw*180/np.pi,npitch*180/np.pi,nroll*180/np.pi,None,None)
        cv2.imwrite(debugfile,img)
        cnt+=1
    return
def AFLWEnhaunceDownScale():
    [outDir]=basFunc.GetCurDirNames(["val"])
    debugDir=basFunc.MakeEmptyDir("debugDir")
    cnt=0
    allfiles=[]
    for file,d,name,ftr in basFunc.GetFileDirLst("AFLW2000",False,True):
        allfiles.append([file,d,name,ftr])
    allcnts=len(allfiles)*2
    for i in range(int(allcnts)):
        basFunc.Process(i,int(allcnts))
        file,d,name,ftr=allfiles[i%len(allfiles)]
        outfile=os.path.join(outDir,str(cnt)+"_enhdw.jpg")
        matfile=os.path.join(d,name+".mat")
        pt2d=get_pt2d_from_mat(matfile)
        x_min,y_min,x_max,y_max=GetBox(file,pt2d)
        cx,cy,w,h=(x_min+x_max)/2,(y_min+y_max)/2,x_max-x_min,y_max-y_min
        w*=2
        h*=2
        x_min,y_min,x_max,y_max=cx-w/2,cy-h/2,cx+w/2,cy+h/2
        x_min,y_min=max(x_min,0),max(y_min,0)
        img=cv2.imread(file)
        x_max,y_max=min(x_max,img.shape[1]-1),min(y_max,img.shape[0]-1)
        box=[x_min,y_min,x_max,y_max]
        pose = get_ypr_from_mat(matfile)
        pitch = pose[0]# * 180 / np.pi
        yaw = pose[1] #* 180 / np.pi
        roll = pose[2]# * 180 / np.pi        
        img=img[int(y_min):int(y_max),int(x_min):int(x_max)]
        img=EnhanunceCVDown(img)
        cv2.imwrite(outfile,img)
        mat=get_R(-pitch,yaw,-roll)
        tp=draw_axis(img,yaw*180/np.pi,pitch*180/np.pi,roll*180/np.pi,0,0,1)
        label=[pitch,yaw,roll]
        pklfile=os.path.join(outDir,str(cnt)+"_enhdw.pkl")
        dio.SaveVariableToPKL(pklfile,[[],label])
        debugfile=os.path.join(debugDir,str(cnt)+".jpg")
        img=draw_axis(img,yaw*180/np.pi,pitch*180/np.pi,roll*180/np.pi,None,None)
        cv2.imwrite(debugfile,img)
        cnt+=1
    return
if __name__ == '__main__':
    #WLP()
    #WLP2()
    #BIWI()
    AFLW() #2000
    AFLWEnhaunce()#4000
    AFLWEnhaunceGray()#2000
    AFLWEnhaunceRotate()#4000
    AFLWEnhaunceDownScale()#4000