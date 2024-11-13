if __name__=='__main__':
    import os,sys
    curDir=os.path.abspath(os.path.dirname(__file__))
    sys.path.append(os.path.join(curDir,'cvSDK'))
    sdkDir=os.path.join(curDir,'cvSDK')
    import warnings
    warnings.filterwarnings("ignore")
    import cvSDK.BasicUseFunc as basFunc
    import cvSDK.DataIO as dio
    import cvSDK.BasicTorchOperator as basTchOp
    import cvSDK.BasicConfig as basCfg
    def GetDefaultDict():    
        dct=basCfg.GetDefaultDict(None)
        return dct
    cfgfile=os.path.join(os.getcwd(),'trainConfig.json')
    if(not os.path.exists(cfgfile)):
        print('Please Run train.py Twice And then Debug trainMMDetection.py')
        exit()
    config=dio.InitJsonConfig(GetDefaultDict,cfgfile,rewrite=False)
    dctcfg=config
    config=dio.Config(config)
    gpus,_=basTchOp.SetGPUConfig(config)
    config=basCfg.ChangeMMDetectionDir(config,dctcfg,cfgfile)
    import mmdet as det
    import cvSDK.MMDetectionEx as mmdetEx
    import cvSDK.BasicMMDet as basDet
    import mmyolo as yolo
    import cvSDK.BasicMMCocoYoloDataSet
    basTchOp.SetTorchSeed(config.seed)
    modelurl=config.modelurl
    if(modelurl==''):
        print('-----------warning--------------')
        print('Please Set trainConfig modelurl ')
        print('-----------warning--------------')
        exit()
    basefile=config.configpyfile
    detDir=basFunc.GetCurrentFileDir(det.__file__)
    detCfgDir=os.path.join(os.path.dirname(detDir),'configs')
    pthfile,pyfile,_=basDet.InitPthandConfig(modelurl,detCfgDir,config.modelpyfile)
    workDir=os.path.join(curDir,config.work_dir)
    basDet.WriteCurrentProcess(workDir)
    def GetDefWeightFile():
        weightFile=''
        _,name,ftr=basFunc.GetfileDirNamefilter(pthfile)
        tpWFile=os.path.join(os.path.join(curDir,'FineTune'),name+ftr)
        print('Check Default File:',tpWFile)
        if(os.path.exists(tpWFile)):
            return tpWFile
        return weightFile
    def ReplaceGetArgsConfig():
        args=basCfg.GetArgsFromTrainConfig(dctcfg)
        args.work_dir=os.path.abspath(args.work_dir)
        file=os.path.basename(args.config)
        args.config=os.path.join(args.work_dir,file)
        return args
    def GetArgsConfig():
        args=basCfg.GetArgsFromTrainConfig(dctcfg)
        return args
    def TrainProcess():
        defFile=GetDefWeightFile()
        if(config.weightFile=='' and defFile!=''):
            config.weightFile=defFile
        if(not os.path.exists(config.weightFile)):
            config.weightFile=''
        global dctcfg
        dctcfg=vars(config)
        import mmdetection.tools.train as basTrain
        curargs=GetArgsConfig()
        cfg=basDet.DumpWorkDirConfig(curargs)
        cfg,dctcfg=basDet.ChangeCfgTrain(cfg,gpus,config,dctcfg)
        cfg.dump(os.path.join(curargs.work_dir, os.path.basename(curargs.config)))
        basCfg.CheckFineTuneDir(config,curDir)
        basTrain.parse_args=ReplaceGetArgsConfig
        basTrain.main()
        return
    TrainProcess()
