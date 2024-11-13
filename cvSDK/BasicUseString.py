import os,sys
import BasicUseFunc as basFunc
def FindNext(txt,left,right,dex):
    lt=txt.find(left,dex)
    rt=txt.find(right,dex)
    if lt<0 and rt <0:return None
    if lt<0:return False,rt
    if rt<0:return True,lt
    if lt<rt:
        return True,lt
    return False,rt
def MatchKuoHao(text,start,left='[',right=']'):
    txt=text[start:]
    cnt=1
    dex=txt.find(left)
    start=dex
    end=None
    while(cnt!=0):
        addflag,dex=FindNext(txt,left,right,dex+1)
        if addflag is None:break
        if addflag:
            cnt+=1
        else:
            cnt-=1
        end=dex
    if end is None: return None,None
    return txt[start:end+1]