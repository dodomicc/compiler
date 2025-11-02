import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from manimlib import *
from utils import *
from typing import *
import numpy as np
from text import *
import cv2
from concurrent.futures import ThreadPoolExecutor



class Basic2DScene(Scene):
    def __init__(
        self,
        frame_callback: Callable[[Scene], None],
        store_key: str = "default",
        args = None,
        **kwargs,
    ):
        self._frame_callback = frame_callback
        self._store_key = store_key
        self._frame_num = 0
        self._args0 = args
        super().__init__(**kwargs)
    def construct(self:Scene):

        self._frame_callback(self)
        
def concat_row(images):
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_img = Image.new('RGBA', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_img.paste(im, (x_offset, 0))
        x_offset += im.width
    return new_img
       
def set_alpha(img: Image.Image) -> Image.Image:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    pixels = img.load()  # 获取像素访问对象
    width, height = img.size
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if r < 100:
                pixels[x, y] = (r, g, b, 0)
    return img
      
        

def generate_manim2d_frames(
    callback_func: Callable[[Scene], None],
    store_key: str = "default",
    args = None
)->Scene:

    scene = Basic2DScene(
        frame_callback=callback_func,
        store_key=store_key,
        args = args
    ) 
    scene.run()
    return scene

def record_Mobject(obj:Mobject,scene:Scene,width:float|None=None,height:float|None=None):
    image_recoder = Group()
    image_recoder._scene = scene
    image_recoder._obj = obj
    image_recoder._frame_idx = 0
    image_recoder._width = width
    image_recoder._height = height
    def image_recoder_updater(mob):
        img = mob._scene.camera.get_image()
        obj_width = mob._obj.get_width()
        obj_height = mob._obj.get_height()
        if(mob._height!=None):
            obj_height = mob._height * 1.1
        if(mob._width!=None):
            obj_width = mob._width * 1.1
        total_width = mob._scene.camera.frame.get_width()
        total_height = mob._scene.camera.frame.get_height()
        left = (total_width-obj_width)/total_width*0.5 * 1920
        top = (total_height-obj_height)/total_height*0.5 * 1080
        right = 1920 - left
        down = 1080 - top
        cropped_image = set_alpha(img.crop((left,top,right,down)))
        if(mob._frame_idx>0):
            img_saver(cropped_image,mob._scene._store_key,mob._frame_idx-1)
            print(f'✅ 成功保存"{mob._scene._store_key}"第{mob._frame_idx}帧')
             
        mob._frame_idx+=1
    image_recoder.add_updater(image_recoder_updater)
    scene.add(image_recoder)
    



def load_image(name:str):
    source_dir = find_nearest_dir_with("assets")
    source_file_path = os.path.join(source_dir,f"{name}.png")
    img = Image.open(source_file_path)
    img_saver(img,name,0)

def get_image(name:str)->Image.Image:
    path = get_frame_path_str(name)
    return Image.open(path)
    

def save_frame_at_time(cap:cv2.VideoCapture,video_name:str,target_time:float):

    target_frame_index = np.round(target_time * 30)
    output_dir = os.path.join(find_nearest_dir_with("assets"),f"common/{video_name}")
    output_image_path = os.path.join(output_dir,f"{video_name}*-*{int(target_frame_index)}.png") 
    # 打开视频
    # 获取视频帧率（FPS）
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 计算目标帧号
    target_frame = int(target_time * fps)
    # 跳转到目标帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    # 读取该帧
    success, frame = cap.read()
    if success:
        os.makedirs(output_dir,exist_ok=True)
        cv2.imwrite(output_image_path, frame)
        print(f'✅ 成功保存"{video_name}"第{target_frame+1}帧')
    else:
        print("❌ 提取失败，请检查时间是否超过视频长度")
    # 释放资源
    
def worker(video_path, video_name, start_t, end_t):
    cap = cv2.VideoCapture(video_path)  # 每个线程自己的cap
    times = np.arange(start_t, end_t, 1/30)
    for t in times:
        save_frame_at_time(cap, video_name, t)
    cap.release()

def load_video(video_name:str,start_at: float=0,end_at:float=1,executor: ThreadPoolExecutor|None= None):
    
    video_path = os.path.join(find_nearest_dir_with("assets"), f"{video_name}.mp4")
    total_length = end_at - start_at
    segment_length = total_length / 8

    segments = [(start_at + i*segment_length, start_at + (i+1)*segment_length) for i in range(8)]

   
        
    for seg_start, seg_end in segments:
        executor.submit(worker, video_path, video_name, seg_start, seg_end)
    

    

    
def get_mobject_textures(content:List[Mobject],width :float= 720,height :float= 720 * 1920/1080):
    def write_simple_tex_callback(scene: Scene):
        scene._res = []
        mob = VGroup()
        def mob_updater(mob0):
            t = scene.time
            if(t-np.floor(t)>0.5 and t -np.floor(t)<0.6):
                index = np.floor(t) + 1
                if(len(scene._res)<index):
                    scene._res.append(scene.camera.get_image())
            else:
                return
        mob.add_updater(mob_updater)
        scene.add(mob)
        
        
        for entry in content:
            scene.add(entry)
            scene.wait(1)
            scene.remove(entry)


    scene = generate_manim2d_frames(
        callback_func=write_simple_tex_callback,
    )
    
    for i, entry in enumerate(scene._res):
        left = int((1920 - width)/2)
        top = int((1080 - height)/2)
        right = 1920 - left
        down = 1080 - top
        entry = entry.crop([left,top,right,down])
        scene._res[i] = entry
        
    res = concat_row(scene._res)
    res = set_alpha(res)

    return res

def load_texture(mob:Mobject,name:str="name"):
    tex = mob
    width = 1920 * (tex.get_width()/14.2222222)
    height = 1080 * (tex.get_height()/8)
    texture = get_mobject_textures( [mob],width=width,height=height)
    img_saver(texture,name)
    

