#%%
import sys 
sys.path.append(r"C:\Users\USER\Desktop\LAO\Image_Recognition\yoloworld_gradio")
from utils.yolo import run_image,setup_runner 
from utils.class_name import class_name
# from utils.efficientsam import load, inference_with_boxes
import supervision as sv
import gradio as gr
import os
import cv2
import sys 
from utils.efficientsam import load, inference_with_boxes,show_anns
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
class_names = class_name()
runner = setup_runner()
# sv.plot_image(img)

def SAM_xy(xy):
    print(xy)
    print('--------')
    for i in xy:
        for j in range(4):
            print(i[j])

# 圖片示例
image_examples = [r"C\Users\USER\Pictures\dog-and-cat-cover.jpg", 2, []]
with gr.Blocks() as app:
    with gr.Row():
        gr.Markdown('''# YOLO-World +Efficient SAM''')
    with gr.Row():
        input_image = gr.Image(type="numpy")
        output_image = gr.Image(label="output")
    with gr.Column():
        input_text = gr.Text(label="輸入字串")
        output_button = gr.Button("輸出")
        radio = gr.Radio(['YES', 'NO'], label='是否使用EfficientSAM遮罩')


    def exe(input_image, input_text,radio):
        if not input_text:
            input_text=class_name()
            
        rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        # 将 RGB 转回 BGR（确保保存的图像颜色正常）
        output_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite("temp_file.jpg", output_image)
        img ,xy = run_image(input_text,runner,"temp_file.jpg")
        if radio=='YES' or not radio:
            mask=inference_with_boxes(input_image,xy)
            mask=show_anns(mask) #生成遮罩照片
            # SAM_xy(xy)     
            # print(mask)
    ################################
            cv2.imwrite("img.jpg", img)
            # cv2.imwrite("mask.png", mask)
            mask = (mask * 255).astype(np.uint8)
            mask = Image.fromarray(mask, 'RGBA')
            output_image_path = 'mask_background.png'
            mask.save(output_image_path)

            img1=cv2.imread('img.jpg')
            img2=cv2.imread('mask_background.png')
            dst=cv2.addWeighted(img1,0.7,img2,0.3,0)
        else:
            return img
##################################
        return dst 

    output_button.click(
            exe,
            inputs=[input_image, input_text,radio],
            outputs=[output_image],
        )

app.launch()

 # %%
# %%
