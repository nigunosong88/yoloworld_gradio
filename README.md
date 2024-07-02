# yoloworld_gradio

# 系統說明
本系統基於Gradio和yoloworld和EfficientSAM模型實現語義分割功能。Gradio提供了一個簡單易用的界面，用戶可以通過該界面輕鬆上傳圖像並查看分割結果。yoloworld則作為底層模型，用於實現對圖像使用文字進行分類，更添加了EfficientSAM實現語義分割。
# 作品展示
## 初始畫面
![image](https://github.com/nigunosong88/yoloworld_gradio/blob/main/images/1.jpg)

## 基礎使用畫面
![image](https://github.com/nigunosong88/yoloworld_gradio/blob/main/images/2.jpg)

## 無使用EfficientSAM基礎使用畫面
![image](https://github.com/nigunosong88/yoloworld_gradio/blob/main/images/2nosam.jpg)

## 無使用EfficientSAM yoloworld 輸入dog 基礎使用畫面
![image](https://github.com/nigunosong88/yoloworld_gradio/blob/main/images/dog.jpg)

## 使用EfficientSAM yoloworld 輸入dog 基礎使用畫面
![image](https://github.com/nigunosong88/yoloworld_gradio/blob/main/images/dogsam.jpg)



# 專案運行方式
安裝環境後，確認文件內容路徑無誤，即可run app.py
# 資料夾說明
- utils
  - 主要架構 yoloworld 與 EfficientSAM
- images
  - 放置實際操作畫面
- yolo_world
  - yolo_world官方其他副程式
# 使用技術
# 主要庫版本
Gradio: 4.26.0  \
EfficientSAM: 版本未指定 \
Segment Anything  \
Yoloworld: 版本未指定 
# 使用版本
python=3.8 \
cuda=11.7 \
gradio=4.16.0 \
inference-gpu=0.9.16 \
mmcv=2.0.0 \
mmdeploy=0.14.0 \
mmengine=0.10.3 \
mmyolo=0.6.0 \
pytorch=1.12.1 \
supervision=0.20.0 
