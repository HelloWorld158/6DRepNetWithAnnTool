import BasicUseFunc as basFunc
import os,sys
import DataIO as dio
import BasicMMDet as basDet

mode='gather'
mode=1
count=0
def GatherHttps(file,dct):
    fp=open(file,'r')
    txts=fp.read()
    fp.close()
    start=0
    end=0
    global count
    while(start!=-1):
        start=txts.find('[model]',start)
        if(start!=-1):start+=8
        else:break
        end=txts.find('&#124;',start)
        if(start!=-1 and end!=-1):
            txt=txts[start:end-2]
            dct[str(count)]=txt
            count+=1
        start=end
    return
if(mode=='gather'):
    import mmdet as det
    mmdetDir=os.path.dirname(det.__file__)
    mmdetDir=os.path.dirname(mmdetDir)
    detDir=os.path.join(mmdetDir,'configs')
    files=[]
    files=basFunc.get_filelist(detDir,files,'.md')
    dct={}
    for file in files:
        GatherHttps(file,dct)
    dio.writejsondictFormatFile(dct,os.path.join(os.getcwd(),'files.json'))
else:
    dct=dio.getjsondata(os.path.join(os.getcwd(),'files.json'))
    count=0
    [basDir]=basFunc.GenerateEmtyDir(['weights'])
    for kk,vv in dct.items():
        print(kk,':Download ',vv)
        pthfile,pyfile=basDet.GetPthFileName(vv)
        basFunc.DownloadFile(vv,pthfile)
    
    
