import os,sys
import os,sys
curDir=os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curDir,'cvSDK'))
sdkDir=os.path.join(curDir,'cvSDK')
import BasicUseFunc as basFunc
import DataIO as dio
import BasicTorchOperator as basTchOp
import shutil
import cvSDK.BasicConfig as basCfg
def GetDefaultDict():    
    dct=basCfg.GetDefaultDict(None)
    return dct
cfgfile=os.path.join(os.getcwd(),'trainConfig.json')
config=dio.InitJsonConfig(GetDefaultDict,cfgfile)
dctcfg=config
config=dio.Config(config)
import cvSDK.BasicMMDet as basDet
workDir=os.path.join(curDir,config.work_dir)
procs=basDet.GetProcess(workDir)
import psutil
pids = psutil.pids()
procNum=len(procs)
while(True):
    for pid in pids:
        p = psutil.Process(pid)
        ppid=p.ppid()
        if ppid in procs and p.pid not in procs:
            procs.append(p.pid)
    if procNum==len(procs):
        break
    procNum=len(procs)
cmd='kill '
for m in procs:
    cmd+=str(m)+' '
print('cmd:',cmd)
os.system(cmd)
basDet.GenerateProcess(workDir)