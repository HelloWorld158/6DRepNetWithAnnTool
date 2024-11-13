### 注意
原始项目地址https://github.com/thohemp/6DRepNet.git<br>
本项目是基于原始项目进行修改,可以无缝支持https://github.com/HelloWorld158/HeadPoseAnnotation.git 这个项目的工具,并稍微修改了模型结构,添加了一个channel负责预测能产生正确旋转量的置信度,如果置信度很低的时候,这个旋转量是不置信的<br>
### 安装
pip install -r requirements.txt<br>
### 训练
Download pre-trained RepVGG model '**RepVGG-B1g2-train.pth**' from [here](https://drive.google.com/drive/folders/1Avome4KvNp0Lqh2QwhXO6L5URQjzCjUq)<br>
将这个预训练权重放到models文件夹下<br>
将https://github.com/HelloWorld158/HeadPoseAnnotation.git 这个工具产生的标注文件分好训练验证分别放到headDir/train 与headDir/val文件夹下<br>
python sixdrepnet/train.py --num_epochs 100 --batch_size 8 train.py的代码相对简单可以直接修改<br>
### 推理
将https://github.com/open-mmlab/mmdetection.git 与 https://github.com/open-mmlab/mmyolo.git 这俩文件内容分别放到 MMDetecTVT/mmdetection 与 MMDetecTVT/mmyolodet <br>
将jpg图片放到valTestData下面然后执行python inferwithdetect.py <br>