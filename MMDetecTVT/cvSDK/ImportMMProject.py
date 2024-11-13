import os,sys
import cvSDK.BasicUseFunc as basFunc
import shutil
import cvSDK.DataIO as dio
import importlib
def CheckfileDir(file,excfile,excdir):
    flag=False
    for exfile in excfile:
        if file.find(exfile)>=0:
            flag=True
            break
    if flag:return True
    for exdir in excdir:
        if file.find(exdir)>=0:
            flag=True
            break
    return flag
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
def CheckMainCode(txt):
    pos=txt.find('from ')
    if pos>=0:return True
    pos=txt.find('import ')
    if pos>=0:return True
    return False
def CheckDouAs(doupos,simpos):
    if(doupos<0 and simpos<0):
        return None,-1
    if doupos>=0 and simpos>=0:
        assert(doupos!=simpos)
        if doupos<simpos:
            return 'dou',doupos
        else:
            return 'as',simpos
    if doupos<0:
        return 'as',simpos
    if simpos<0:
        return 'dou',doupos
    raise NotImplementedError('not implementerror')


def GetSplitText(txt):    
    pos=txt.find('from ')
    if pos>=0:
        impos=txt.find(' import')
        midtxt=txt[pos+len('from '):impos]
        return [txt[:pos+len('from ')],midtxt,txt[impos:]],[0,1,0]
    pos=txt.find('import ')
    txts=[txt[:pos+len('import ')]]
    flags=[0]
    pos=len(txts[-1])
    impos=txt.find(' as ',pos)
    doupos=txt.find(',',pos)
    if impos<0 and doupos<0:
        txts.append(txt[pos:])
        return txts,[0,1]    
    while(pos>=0):
        doupos=txt.find(',',pos)
        simpos=txt.find(' as ',pos)
        stat,dspos=CheckDouAs(doupos,simpos)
        if stat is None:
            if txts[-1]==',':
                flags.append(1)
            else:
                flags.append(0)
            txts.append(txt[pos:])            
            break
        length=0
        if stat=='dou':
            length=len(',')
        else:
            length=len(' as ')
        txts.append(txt[pos:dspos])
        txts.append(txt[dspos:dspos+length])
        flags.extend([1,0])
        pos=dspos+length
    return txts,flags
def CheckTxtDct(kk,t):
    pos=kk.find(t)
    if pos<=0:
        return False
    if pos+len(t)<len(kk):
        return False
    if pos==0:
        return True
    if kk[pos-1]!='.':
        return False
    return True
def GetFinalStr(kk,txts,dex,flags):
    if txts[dex].find('.')>=0:
        return kk
    if len(txts)>dex+1 and txts[dex+1]==' as ':
        return kk
    if len(txts)>dex+1 and txts[dex+1].find('import')>0 and flags[dex+1]==0:
        return kk
    pos=0
    while(' '==txts[dex][pos]):
        pos+=1
    t=txts[dex][pos:]
    return kk+' as '+t
def ExcludeSpecText(txt):
    pos=txt.find('#')
    if pos>=0:
        curtxt=txt[:pos]
        leftxt=txt[pos:]
    else:
        curtxt=txt
        leftxt=''
    txts,flags=GetSplitText(curtxt)
    return txts,flags,leftxt
def FindImportTxt(dct,txt,curkey):
    lst=[]
    txts,flags,spectxt=ExcludeSpecText(txt)    
    for i,t in enumerate(txts):
        if flags[i]==0:
            lst.append([])
            continue
        if t=='.':
            lst.append([curkey])
            continue
        mlst=[]
        for kk,vv in dct.items():    
            if CheckTxtDct(kk,t):
                mlst.append(GetFinalStr(kk,txts,i,flags))
        lst.append(mlst)    
    txts[-1]+=spectxt            
    return txts,flags,lst
def ExchangeOriTxts(txtes,flags,lst):
    txt=''
    for i,t in enumerate(txtes):
        if flags[i]==0:
            txt+=t
            continue
        txt+=lst[i][0]
    return txt
def ExchangeDictImport(dct,txts,curfile,curkey):
    flag=False
    for i,txt in enumerate(txts):
        txt=txt[:-1]
        if not CheckMainCode(txt):
            continue
        txtes,flags,lst=FindImportTxt(dct,txt,curkey)
        curflag=True
        for j,t in enumerate(txtes):
            if len(lst[j])>0:
                curflag=False
        if curflag:
            continue
        curflag=False
        for j,t in enumerate(txtes):
            if len(lst[j])>=2:
                print(curfile,' dex:',i,'out',lst[j])
                curflag=True
        if  curflag:
            continue        
        txt=ExchangeOriTxts(txtes,flags,lst)
        txts[i]=txt+'\n'
    return flag,txts
def ChangeBackUpOriFile(dct):    
    cnt=0
    for kk,vv in dct.items():
        cnt+=1
        basFunc.Process(cnt,len(dct))
        file=vv['filename']
        fp=open(file,'r')
        txts=fp.readlines()
        fp.close()
        flag,txts=ExchangeDictImport(dct,txts,file,kk)
        if flag:continue
        fp=open(file,'w')
        fp.writelines(txts)
        fp.close()
    return
def ChangeImportFiles(projectfiles):
    profiles=[]
    copyflag=False
    for file in projectfiles:
        d,name,ftr=basFunc.GetfileDirNamefilter(file)
        newfile=os.path.join(d,name+'Import'+ftr)
        profiles.append(newfile)
        if os.path.exists(newfile):continue
        copyflag=True
        shutil.copy(file,newfile)
    return profiles,copyflag
def ChangeDirRelateImport(chgdir):
    upDir=os.path.dirname(chgdir)
    projectDirs=basFunc.GetCurDirNames(['cvExtendModel','cvSDK'],chgdir)
    projectfiles=basFunc.GetCurDirNames(['ExportMMProject.py'],chgdir)
    projectfiles,copyflag=ChangeImportFiles(projectfiles)
    if not copyflag:
        print(f'find already convert absimport if you want change please delete {projectfiles[0]}')
        return
    files=[]    
    for proj in projectDirs:
        files+=basFunc.get_filelist(proj,files,'.py')
    files+=projectfiles    
    dct={}
    for file in files:
        dirs=basFunc.GetRelativeRootFile(file,upDir)
        imp=GetImportString(dirs)
        dct[imp]={'filename':file}
    ChangeBackUpOriFile(dct)
    return
curDir=os.path.dirname(basFunc.GetCurrentFileDir(__file__))
jsdata=dio.getjsondata(os.path.join(curDir,'trainConfig.json'))
jsdata=dio.Config(jsdata)
cfgfile=jsdata.configpyfile
if not os.path.isabs(cfgfile):
    cfgfile=os.path.join(curDir,cfgfile)
pyfile=cfgfile[len(curDir)+1:].replace('/','.')[:-3]
exec(f'import {pyfile} as fn')
assert(hasattr(fn,'custom_imports') and hasattr(fn,'import_project'))
projects=fn.import_project
for project in projects:
    project=os.path.abspath(project)
    upDir=os.path.dirname(project)
    basName=os.path.basename(project)
    os.environ['extendDir']=upDir
    if upDir not in sys.path:
        sys.path.append(upDir)
    ChangeDirRelateImport(project)
    cmd=f'{basName}.ExportMMProjectImport'
    importlib.import_module(cmd)
    print(f'success import projet {project}')
print('end project imports')