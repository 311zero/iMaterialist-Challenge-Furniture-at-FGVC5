# iMaterialist-Challenge-Furniture-at-FGVC5
kaggle比赛‘iMaterialist Challenge(Furniture)at FGVC5’家居识别比赛，是一个多分类问题，128类  

将照片中的产品自动分类。难点在于同一个产品，可以在不同的灯光、角度、背景、遮挡物下拍摄；同时，不同细粒度的类别看起来可能非常相似；而且图像大小从265到5792像素不等。  
## 使用  
从kaggle里面下载数据集：https://www.kaggle.com/c/imaterialist-challenge-furniture-2018 下载到`./data/`    
下载图片使用: python downloader.py  
训练模型使用：python cnn_runner.py train  
使用训练好的模型进行预测：python cnn_runner.py predict  
生成提交的文件：python submit.py  
### 结果 
目前排名：76/428 分数：0.14822
