import os,sys
curDir=os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curDir,'cvSDK'))
sdkDir=os.path.join(curDir,'cvSDK')
import cvSDK.BasicUseFunc as basFunc
import cvSDK.DataIO as dio
import cvSDK.BasicTorchOperator as basTchOp
import shutil
import cvSDK.BasicConfig as basCfg
workDir=os.path.join(curDir,'workDir')
def GetDefaultDict():    
    dct=basCfg.GetDefaultDict(None)
    return dct
cfgfile=os.path.join(os.getcwd(),'trainConfig.json')
if(not os.path.exists(cfgfile)):
    print('Please Run train.py Twice And then Debug trainMMDetection.py')
    exit()
config=dio.InitJsonConfig(GetDefaultDict,cfgfile)
dctcfg=config
config=dio.Config(config)
gpus,_=basTchOp.SetGPUConfig(config)
config=basCfg.ChangeMMDetectionDir(config,dctcfg,cfgfile)
import mmdet as det
import mmdet.utils as detutils
detutils.register_all_modules(False)
import cvSDK.MMDetectionEx as mmdetEx
import cvSDK.BasicMMDet as basDet
import mmyolo as yolo
import mmyolo.utils as yoloutils
yoloutils.register_all_modules(False)
import cvSDK.BasicMMCocoYoloDataSet
from mmengine import Config
from mmengine.dist import is_distributed
basTchOp.SetTorchSeed(config.seed)
modelurl=config.modelurl
basefile=config.configpyfile
detDir=basFunc.GetCurrentFileDir(det.__file__)
detCfgDir=os.path.join(os.path.dirname(detDir),'configs')
pthfile,pyfile,_=basDet.InitPthandConfig(modelurl,detCfgDir,config.modelpyfile)
def GetArgsConfig():
    global pthfile,gpus
    args=basCfg.GetArgsFromTestConfig(dctcfg,pthfile,gpus)
    d,name,ftr=basFunc.GetfileDirNamefilter(args.config)
    pyconfig=Config.fromfile(basefile)
    dataloaders=['train_dataloader','val_dataloader','test_dataloader']
    for loader in dataloaders:
        curloader=eval(f'pyconfig.{loader}')
        if curloader.num_workers==0:
            exec(f'pyconfig.{loader}.persistent_workers=False')
    pyconfig.dump(os.path.join(workDir,name+ftr))
    args.config=os.path.join(workDir,name+ftr)
    if not is_distributed():
        args.launcher='none'
    return args
def TestProcess():
    import mmdetection.tools.test as basTest
    global pthfile
    pthfile=config.TestWeightFile
    pthfile=basCfg.GetDefaultBestPth(curDir,config,pthfile)
    print('UseTrainPth:',pthfile)
    basTest.parse_args=GetArgsConfig
    basTest.main()
    return
print('start Test...^_^,GoodLuck')
TestProcess()
