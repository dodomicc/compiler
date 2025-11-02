


import os
import re
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips


def merge_mp4_mp3(folder='.'):
    """
    批量合并文件夹中同名的 mp4 和 mp3 文件，
    把 mp3 作为音频替换到 mp4 中，
    输出文件名为 “原文件名-合并.mp4”。
    """
    for filename in os.listdir(folder):
        if filename.endswith('.mp4'):
            base = filename[:-4]
            mp3_file = os.path.join(folder, base + '.mp3')
            mp4_file = os.path.join(folder, filename)
   
            if os.path.exists(mp3_file):
                video = VideoFileClip(mp4_file)
                audio = AudioFileClip(mp3_file)
                output_video = video.with_audio(audio)
                output_video.write_videofile(f"{base}-声音画面合成版本.mp4")
                print(f'合并 {filename} 和 {base}.mp3 → f"{filename}-声音文字合成版本.mp4"')
                
def extract_title(filename):
    """
    从 'title-声音画面合成版本.mp4' 中提取 title 部分
    """
    match = re.match(r'(.*)-声音画面合成版本\.mp4$', filename)
    return match.group(1) if match else None
                
def merge_videos_by_title(source_dir = './', output_path='合成视频.mp4'):
    """
    合并指定目录下所有形如 'title-声音画面合成版本.mp4' 的视频文件，
    并按 title 排序后拼接生成一个视频。
    """

    # 过滤并提取文件
    video_files = []
    for file in os.listdir(source_dir):
        if file.endswith('声音画面合成版本.mp4'):
            title = extract_title(file)
            if title:
                video_files.append((title, os.path.join(source_dir, file)))

    if not video_files:
        print("❌ 没有找到符合命名规则的视频文件。")
        return

    # 按 title 排序
    video_files.sort(key=lambda x: x[0])

    # 加载所有视频剪辑
    clips = []
    for title, path in video_files:
        print(f'📦 加载视频：{path}')
        clip = VideoFileClip(path)
        clips.append(clip)

    # 合并视频
    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

    print(f'✅ 合成完成：{output_path}')

if __name__ == "__main__":
    merge_mp4_mp3()
    #merge_videos_by_title()