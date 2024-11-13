import os,sys
import BasicUseFunc as basFunc
import BasicCocoFunc as basCoco
mode=['OWNXMLDATA']#'Initial','VOC','OBJECT365','OWNXMLDATA','OWNJSONDATA','COCO'
crtCoco=True
curDir=os.getcwd()
def BeginWord():
    curFile=os.path.join(curDir,'ConvertDataSet.py')
    fp=open(curFile,'r')
    txts=fp.readlines()
    linDex,curLineDex=basFunc.Findtxtlinelst(txts,'ResetMode=')
    txts[curLineDex]='ResetMode=True\n'
    fp=open(curFile,'w')
    fp.writelines(txts)
    fp.close()
    print('--------------------------------------------')
    print('--------------------------------------------')
    print('Enable ConvertDataSet.py File ResetMode:True')
    print('--------------------------------------------')
    print('--------------------------------------------')
    return
def ReadSavePython(m):
    fp=open(os.path.join(os.getcwd(),'ConvertDataSet.py'),'r')
    txtlines=fp.readlines()
    fp.close()
    linDex,curLineDex=basFunc.Findtxtlinelst(txtlines,'mode=\'')
    txtlines[curLineDex]='mode=\''+m+'\'\n'
    if(crtCoco):
        linDex,curLineDex=basFunc.Findtxtlinelst(txtlines,'bConvertOthData')
        txtlines[curLineDex]='bConvertOthData=True\n'
    fp=open(os.path.join(os.getcwd(),'ConvertDataSet.py'),'w')
    fp.writelines(txtlines)
    fp.close()
BeginWord()
for m in mode:
    #break
    ReadSavePython(m)
    print('curProcess:',m)
    sys.stdout.flush()
    os.system('python3 ConvertDataSet.py')

curDir=os.getcwd()
trainDir=os.path.join(curDir,'train')
validDir=os.path.join(curDir,'valid')
if(crtCoco):
    basCoco.ConvertCoCoDataSet(trainDir)
    basCoco.ConvertCoCoDataSet(validDir)

