#%%
from yolo_world.yolo import run_image,setup_runner 
from class_name import class_name
import supervision as sv
import gradio as gr
import os
import cv2
import sys 
   
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
        cv2.imwrite("temp_file.jpg", input_image)
        img ,xy = run_image(input_text,runner,"temp_file.jpg")
        SAM_xy(xy)
        # if radio=='YES' or not radio:
            
        
        return img

    output_button.click(
            exe,
            inputs=[input_image, input_text,radio],
            outputs=[output_image],
        )

app.launch()

# %%
