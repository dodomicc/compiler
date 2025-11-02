
import subprocess
import re

import os
import shutil

def is_english(content: str) -> bool:
    import string

    total_chars = len(content)
    if total_chars == 0:
        return False  # 空内容默认判定为非英文

    english_letters = sum(1 for c in content if c in string.ascii_letters)
    return english_letters / total_chars > 0.5

def ensure_and_clear_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"✅生成文件夹: {folder_name}")
    else:
        for filename in os.listdir(folder_name):
            file_path = os.path.join(folder_name, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # 删除文件或符号链接
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # 删除子文件夹
                print(f"✅删除文件成功: {file_path}")
            except Exception as e:
                print(f"❌删除 {file_path}失败. Reason: {e}")



# 语音配置（可选语音参考下方列表）


def extract_sections_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 正则匹配：以 # 开头，提取两个 $...$ 对的内容
    pattern = r"#\s*\$(.*?)\$\s*\$(.*?)\$"
    matches = re.findall(pattern, text, re.DOTALL)

    # 转为结构化数组
    sections = [{"title": title.strip(), "content": content.strip()} for title, content in matches]

    return sections

def text_to_speech(text, voice, filename):
    cmd = [
        "edge-tts",
        "--proxy", "",

        "--voice", voice,
        "--text", text,
        "--write-media", f"./audios/audio-{filename}.mp3",
         "--write-subtitles", f"./subtitles/subtitles-{filename}.vtt"
   
    ]
    subprocess.run(cmd)
    print(f"✅已生成视频文件 audios/audio-{filename}.mp3 文件")
    print(f"✅已生成字幕文件 subtitles/subtitles-{filename}.vtt 文件")
    
def copy_audio(base_dir = './'):
    """
    从 base_dir/audio 中提取所有 audio-xxx.mp3 文件，
    将它们复制到 ../视频声音合并 目录，重命名为 xxx.mp3。
    """

    source_audio_dir = os.path.join(base_dir, 'audios')
    target_dir = os.path.abspath(os.path.join(base_dir, '..', '视频声音合并'))
    os.makedirs(target_dir, exist_ok=True)

    for filename in os.listdir(source_audio_dir):
        if filename.endswith('.mp3') and filename.startswith('audio-'):
            # 提取 title 部分
            title = filename[len('audio-'):-len('.mp3')]
            source_path = os.path.join(source_audio_dir, filename)
            target_path = os.path.join(target_dir, f'{title}.mp3')

            shutil.copy2(source_path, target_path)
            print(f'✅ 已复制：{filename} → {title}.mp3')

if __name__ == "__main__":

    ensure_and_clear_folder("audios")
    ensure_and_clear_folder("subtitles")
    sections = extract_sections_from_file("./script/script.txt") 
    
    for section in sections:
        title = section["title"]
        content = section["content"]
        contenr = content.encode("utf-8", errors="ignore").decode("utf-8")
        voice = "en-US-GuyNeural" if is_english(content) else "zh-CN-YunxiNeural" 
        text_to_speech(content, voice, title)
        
    copy_audio()
 




