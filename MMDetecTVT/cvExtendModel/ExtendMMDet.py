import os,sys
import BasicUseFunc as basFunc
curDir=basFunc.GetCurrentFileDir(__file__)
sys.path.append(curDir)
extendDir=os.path.join(curDir,'ExtendDepends')
sys.path.append(extendDir)
files=[]
excludelst=['ExtendDepends','ExtendMMDet.py','.ipynb_checkpoints']
files=basFunc.get_filelist(curDir,files,'.py')
_,curName,_=basFunc.GetfileDirNamefilter(__file__)
names=[]
def CheckAvalibleFile(dirs):
    for dir in dirs:
        if(dir in excludelst):return False
    return True
def GetImportString(dirs):
    txts=''
    for i in range(len(dirs)):
        if(i==0):continue
        txt=''
        if(i==len(dirs)-1):
            txt=dirs[i][:-3]
        else:
            txt=dirs[i]+'.'
        txts+=txt
    return txts
importlst=[]
rootDir=curDir
if curDir.find(os.path.abspath(os.getcwd()))>=0:
    rootDir=os.getcwd()
if 'extendDir' in os.environ:
    rootDir=os.environ['extendDir']
print('ExtendRootDir:',rootDir)
for file in files:    
    dirs=basFunc.GetRelativeRootFile(file,rootDir)
    if(not CheckAvalibleFile(dirs)):continue
    cmdtxt=GetImportString(dirs)
    importlst.append(cmdtxt)
print('Load MMEx Modules:',len(importlst))
for imp in importlst:
    exec('import '+imp)