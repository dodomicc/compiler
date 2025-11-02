import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from manimlib import *
from manim import Wait
from typing import *
from pathlib import Path
import struct

color_list = [WHITE, BLACK, RED, GREEN, BLUE, YELLOW, PURPLE, ORANGE, PINK, TEAL, MAROON, GOLD]
frames_store: Dict[str, int] = {}





#将uv坐标转化为manim坐标
def uvToManimCoord(*input):
    uv = list(input)
    res = [(2 * uv[0] - 1)*7.1111111, (2 * uv[1] - 1)*4, 0]
    return res


def make_find_nearest_dir_with():
    cache = {}

    def find_nearest_dir_with(folder_name: str, start_path: Path = None) -> Path | None:
        nonlocal cache
        if folder_name in cache:
            return cache[folder_name]

        if start_path is None:
            start_path = Path.cwd()  # 默认从当前工作目录开始

        current = start_path.resolve()

        while current != current.parent:
            candidate = current / folder_name
            if candidate.exists() and candidate.is_dir():
                cache[folder_name] = candidate
                return candidate
            current = current.parent

        cache[folder_name] = None
        return None

    return find_nearest_dir_with


# 创建闭包函数对象
find_nearest_dir_with = make_find_nearest_dir_with()


    
def list_files_folder_name(folder_path:Path):
  
    p = Path(folder_path)
    names = []
    
    for name in p.iterdir():
        names.append(name)
    return names
    
def get_frame_path_str(name: str,idx:int=0,isCycled:bool=True) ->str:
    """
    从frames_store中提取指定key的第index帧。
    index从0开始,如果index越界,则返回最后一帧。
    """
    
    dir_path = os.path.join(find_nearest_dir_with("assets"),f"common/{name}")

   
    if(not hasattr(frames_store,name)):
        frames_store[name] = len(list_files_folder_name(dir_path))
    if(frames_store[name]==1):
        file_name = list_files_folder_name(dir_path)[0]
        return os.path.join("assets/common/{name}",file_name)
    
    index = idx%frames_store[name]
    if(not isCycled):
        index = min(idx, frames_store[name]-1)
    path = os.path.join(dir_path,f"{name}*-*{index}.png")
    
    return str(path)

def compute_texture_frame_path_str(name: str,idx:int=0) ->str:
    """
    从frames_store中提取指定key的第index帧。
    index从0开始,如果index越界,则返回最后一帧。
    """
    
    dir_path = os.path.join(find_nearest_dir_with("assets"),f"common/{name}")
    path = os.path.join(dir_path,f"{name}*-*{idx}.png")
    
    return str(path)

def encode_float(num:float)->Tuple:
    return struct.unpack('4B', struct.pack('f', num))

def decode_float(bytes_tuple: Tuple[int, int, int, int]) -> float:
    return struct.unpack('f', struct.pack('4B', *bytes_tuple))[0]







 
def img_saver(img:Image.Image,name:str,frame:float = 0):
    source_dir = find_nearest_dir_with("assets")
    dir_path = os.path.join(source_dir,f"common/{name}")
    os.makedirs(dir_path,exist_ok=True)
    file_path = os.path.join(dir_path,f"{name}*-*{frame}.png")
    img.save(file_path)
    print(f"✅ 成功保存{name}第{frame+1}帧")
    

def image_shower(
            mob0:Mobject,
            name:str,
            start_at:float,
            end_at:float,
            scene:Scene,
            height:float=3,
            isCycled:bool=True
        ):
        file_path = get_frame_path_str(name,0)
        old_image = ImageMobject(file_path,height=height/2).move_to(mob0)
        mob1 = VGroup()
        mob1._old_image = old_image
        mob1._isCycled = isCycled
        mob1._name = name
        mob1._scene = scene
        mob1._start_at =start_at
        mob1._end_at = end_at
        mob1._height = height
        mob1._control_mob = mob0
        mob1._removed = False
        def image_updater(mob):
            t = mob._scene.time
            if(t<mob._start_at):
                return
            elif(t<mob._end_at):
                frameIndex = int((t-mob._start_at)*30)
                image_name:str = mob._name
                file_path = get_frame_path_str(image_name,frameIndex,isCycled= mob._isCycled)
                old_image = mob._old_image
                mob._scene.remove(old_image)
                new_image = ImageMobject(file_path).set_height(mob._height).move_to(mob._control_mob)
                mob._scene.add(new_image)
                mob._old_image = new_image
            else:
                if(not mob._removed):
                    mob._scene.remove(mob._old_image)
                    mob._removed = True
                return 
        mob1.add_updater(image_updater)
        scene.add(mob1)

        


    

def remap(val1,val2,val3,val4,val):
    if(val1 == val2): return val3
    h = (val - val1)/(val2 - val1)
    res = val3 + h * (val4 - val3)
    return res 





def keep_obj_alive(mob:Mobject,start_at:float,end_at:float,scene:Scene):
    m0 = Group()
    m0._mob= mob
    m0._added = False
    m0._removed = False
    m0._scene = scene
    m0._start_at = start_at
    m0._end_at = end_at
    def m0_updater(mob):
        t = mob._scene.time
        if(t<mob._start_at):
            return
        elif t<mob._end_at :
            if(mob._added != True):
                mob._scene.add(mob._mob)
                m0._added=True
        else:
            if(mob._removed != True):
                mob._scene.remove(mob._mob)
                mob._removed=True
    m0.add_updater(m0_updater)
    scene.add(m0)
    
def keep_obj_invisable(mob:Mobject,start_at:float,scene:Scene):
    m0 = Group()
    m0._mob= mob
    m0._added = False
    m0._removed = False
    m0._scene = scene
    m0._start_at = start_at

    def m0_updater(mob):
        t = mob._scene.time
        if(t<mob._start_at):
            return
        else :
            if(mob._removed != True):
                mob._scene.remove(mob._mob)
                mob._removed=True
    m0.add_updater(m0_updater)
    scene.add(m0)
    

    

def keep_obj_alive_update(mob:Mobject,start_at:float,end_at:float,scene:Scene,updater: Callable|None=None):
    m0 = Group()
    m0._mob= mob
    m0._added = False
    m0._removed = False
    m0._scene = scene
    m0._start_at = start_at
    m0._end_at = end_at
    m0._updater = updater
    def m0_updater(mob):
        t = mob._scene.time
        if(t<mob._start_at):
            return
        elif t<mob._end_at :
            if(mob._added != True):
                mob._scene.add(m0._mob)
                m0._added=True
            if(mob._updater!=None):
                mob._updater(mob._mob,t)
            else:
                return
        else:
            if(mob._removed != True):
                mob._scene.remove(m0._mob)
                m0._removed=True
    m0.add_updater(m0_updater)
    scene.add(m0)
    
def add_notation(source_obj:Mobject,dist_obj:Mobject,direction:np.ndarray = RIGHT,buff :float = 0.5):
    if np.all(direction == RIGHT):
        dist_obj.move_to([source_obj.get_center()[0]-source_obj.get_width()/2 - buff,source_obj.get_center()[1],0],aligned_edge=RIGHT)
    if np.all(direction == DOWN):
        dist_obj.move_to([source_obj.get_center()[0],source_obj.get_center()[1]+source_obj.get_height()/2 + buff,0],aligned_edge=DOWN)
    

            
    


