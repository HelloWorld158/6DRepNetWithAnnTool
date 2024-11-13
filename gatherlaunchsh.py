import os,sys
import json
jsfile=os.path.join(os.getcwd(),'.vscode/launch.json')
def getjsondata(data_file):
    with open(data_file) as f:
        txts = f.readlines()
    ntxts=[]
    for txt in txts:
        dex=txt.find('//')
        if dex>=0:
            txt=txt[:dex]
        if len(txt)==0:continue
        ntxts.append(txt.strip())
    finaldata=''.join(ntxts)
    try:
        data = json.loads(finaldata)
    except Exception as e:
        print('error',str(e))
        return None
    return data
jsdata=getjsondata(jsfile)
fp=open('launch.txt','w+')
for dct in jsdata['configurations']:
    name=dct['program']
    oriname=name
    dex=name.find('.py')
    if dex>=0:
        name=name[:dex]
        pyflag=True
    else:
        pyflag=False
    shname=name.replace('/','_')+'.sh'
    if 'args' not in dct:
        dct['args']=[]
    with open(os.path.join(os.getcwd(),shname),'w+') as fp:
        fp.write('#!/bin/bash\n')
        if pyflag:
            fp.write('python '+oriname+' '+' '.join([a for a in dct['args']])+'\n')
        else:
            fp.write(oriname+' '+' '.join([a for a in dct['args']])+'\n')
    
