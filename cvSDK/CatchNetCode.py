import BasicUseFunc as basFunc
import requests
import os,sys
import re
import  time
import  random
def getManyPages(keyword,pages):
    params=[]
 
    for i in range(30,30*pages+30,30):
        params.append({
                      'tn': 'resultjson_com',
                      'ipn': 'rj',
                      'ct': 201326592,
                      'is': '',
                      'fp': 'result',
                      'queryWord': keyword,
                      'cl': 2,
                      'lm': -1,
                      'ie': 'utf-8',
                      'oe': 'utf-8',
                      'adpicid': '',
                      'st': -1,
                      'z': '',
                      'ic': 0,
                      'word': keyword,
                      's': '',
                      'se': '',
                      'tab': '',
                      'width': '',
                      'height': '',
                      'face': 0,
                      'istype': 2,
                      'qc': '',
                      'nc': 1,
                      'fr': '',
                      'pn': i,
                      'rn': 30,
                      'gsm': '1e',
                      '1488942260214': ''
                  })
    url = 'https://image.baidu.com/search/acjson'
 
 
   # regex = re.compile(r'\\(?![/u"])')
  #  new_url = regex.sub(r"\\\\", url)
 
    urls = []
    #for i in params :
       # new_params = regex.sub(r"\\\\", params[i])
 
    for i in params:
#        regex = re.compile(r'\\(?![/u"])')
#        fixed = regex.sub(r"\\\\", params[i])
 
        urls.append(requests.get(url,params=i).json().get('data'))
 
    return urls
 
def getpage(key,page):
    new_url = []
    for i in range(0, page*30+30, 30):
        new_url.append({
            'tn': 'resultjson_com',
            'ipn': 'rj',
            'ct': 201326592,
            'is': '',
            'fp': 'result',
            'queryWord': key,
            'cl': 2,
            'lm': -1,
            'ie': 'utf-8',
            'oe': 'utf-8',
            'adpicid': '',
            'st': '',
            'z': '',
            'ic': '',
            'word': key,
            's': '',
            'se': '',
            'tab': '',
            'width': '',
            'height': '',
            'face': '',
            'istype': '',
            'qc': '',
            'nc': '',
            'fr': '',
            'pn': i,
            'rn': 30,
            'gsm': '3c',
            '1517056200441': ''
        })
 
    url = 'https://image.baidu.com/search/acjson'
    result=[]
 
    for i in  new_url:
        randnumber1 = random.randint(0,3)#生成随机数
        time.sleep(randnumber1)#按随机数延时
        print(i)
        try:
            data=requests.get(url, params=i).json().get('data')
            result.append(data)
            #print(result)
        except :#如果延时之后还是被拒绝
            print('error\n')
            randnumber2 = random.randint(5,10)#延迟随机时间
            time.sleep(randnumber2)
 
 
    #print(result)
 
 
    return result
 
def getImg(dataList, localPath,keyword):
    i=1
    x = 0
    for list in dataList:
        for each in list:
            ###################
            try:
                if each.get('thumbURL') != None:
                    print('downloading:%s' % each.get('thumbURL'))
                    pic = requests.get(each.get('thumbURL'))
            except requests.exceptions.ConnectionError:
                print('error: This photo cannot be downloaded')
                continue
 
            dir = localPath+'/' + keyword + '_' + str(i) + '.jpg'
            fp = open(dir, 'wb')
            fp.write(pic.content)
            fp.close()
            i += 1
 

def downloadBaiduPic(keyword,strdir,maxNum):
    dataList = getpage(keyword,maxNum)  # key word and number of page
    getImg(dataList,strdir,keyword)
def downloadspider(keyword,strdir,maxNum):
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.125 Safari/537.36'}
    name = keyword
    num = 0
    x = maxNum
    i=0
    repeatMaxNum=100
    lastNum=0
    repeatNum=0
    while(True):    
        name_1 = strdir
        url = 'https://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word='+name+'&pn='+str(i*30)
        i+=1
        res = requests.get(url,headers=headers)
        htlm_1 = res.content.decode()
        a = re.findall('"objURL":"(.*?)",',htlm_1)
        if not os.path.exists(name_1):
            os.makedirs(name_1)
        for b in a:
            num = num +1
            try:
                img = requests.get(b)
            except Exception as e:
                print('第'+str(num)+'张图片无法下载------------')
                num=num-1
                print(str(e))
                continue
            f = open(os.path.join(name_1,str(num)+'.jpg'),'wb')
            print('---------正在下载第'+str(num)+'张图片----------')
            f.write(img.content)
            f.close()
            if(num>maxNum):
                return
        if lastNum!=num:
            lastNum=num
            repeatNum=0
        else:
            repeatNum+=1
            if repeatNum>repeatMaxNum:return
        
if __name__ == '__main__':
    keywords=['helloworld']
    maxNums=[]
    defaultnum=1000
    flag=len(keywords)==len(maxNums)
    ndir=os.path.abspath(os.path.dirname(__file__))
    for i in range(len(keywords)):
        print('Process:'+str(i)+'/'+str(len(keywords)))
        curDir=os.path.join(ndir,str(i))
        basFunc.MakeEmptyDir(curDir)
        num=defaultnum
        if flag:
            num=maxNums[i]
        downloadspider(keywords[i],curDir,num)
        



