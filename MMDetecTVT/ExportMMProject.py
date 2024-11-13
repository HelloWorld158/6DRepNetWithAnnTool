'''
backbone=dict(type='mmdet.DarknetYolov5', 
              deepen_factor=0.33, 
              widen_factor=0.5,
              spp_kernel_sizes=5,
              out_indices=( 4,),
              act_cfg=dict(type='SiLU'),
              spp_last=True,
              init_cfg=dict(type='Pretrained', 
                            checkpoint='FineTune/yolov5s_mm.pth', 
                            # yolo_root='/notebooks/zw/projects/yolov5_61'  # 存在时读取官方weight，不使用时读取mm结果权重
                        )),
custom_imports = dict(imports=['cvSDK.ImportMMProject'], allow_failed_imports=False)
import_project=['../MMDetecTVT']
'''
import os,sys
import cvSDK.BasicUseFunc as basFunc
import cvSDK.DataIO as dio
import cvSDK.BasicConfig as basCfg
def GetDefaultDict():
    return basCfg.GetDefaultDict('train')
curDir=os.path.abspath(os.path.dirname(__file__))
cfgdct=dio.InitJsonConfig(GetDefaultDict,os.path.join(curDir,'trainConfig.json'))
config=dio.Config(cfgdct)
basCfg.ChangeMMDetectionDir(config,cfgdct,os.path.join(curDir,'trainConfig.json'))
import cvSDK.BasicMMDet as basDet
detDir=basDet.GetMMDetDir()
sys.path.append(detDir)
import mmdet as det
import mmdet.utils as detutils
detutils.register_all_modules(False)
import cvSDK.MMDetectionEx as mmdetEx
import cvSDK.BasicMMDet as basDet
import mmyolo as yolo
import mmyolo.utils as yoloutils
yoloutils.register_all_modules(False)
import cvSDK.BasicMMCocoYoloDataSet

