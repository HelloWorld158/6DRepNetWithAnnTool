import BasicUseFunc as basFunc
import numpy as np
import sys,os
import DataIO as dio
import shutil
[extDir,tempDir]=basFunc.GenerateEmtyDir(['ExtractDir','tempDir'])
files=[]
files=basFunc.get_filelist(os.getcwd(),files,'.mp4')
count=0
for file in files:    
    basFunc.MakeEmptyDir(tempDir)
    dio.SaveFrame(file,tempDir,iSample=100000)
    count+=1
    #basFunc.Process(count,len(files))
    bfiles=basFunc.getdatas(tempDir)
    for bfile in bfiles:
        _,name,ftr=basFunc.GetfileDirNamefilter(bfile)
        shutil.copy(bfile,os.path.join(extDir,str(count)+'_'+name+ftr))
exit()
for file in files:    
    basFunc.MakeEmptyDir(tempDir)
    dio.SaveFrame(file,tempDir,iSample=10)
    _,namex,_=basFunc.GetfileDirNamefilter(file)
    count+=1
    #basFunc.Process(count,len(files))
    bfiles=basFunc.getdatas(tempDir)
    for bfile in bfiles:
        _,name,ftr=basFunc.GetfileDirNamefilter(bfile)
        shutil.copy(bfile,os.path.join(extDir,namex+'_'+name+ftr))
    