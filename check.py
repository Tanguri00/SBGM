import os
import re

import cv2
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

# 配置路径
DATA_ROOT = Path("data/MSVD")
CAPTION_CSV = DATA_ROOT / "video_corpus.csv"
OFFICIAL_SPLITS = {
    "train": DATA_ROOT / "train_list.txt",
    "val": DATA_ROOT / "val_list.txt",
    "test": DATA_ROOT / "test_list.txt"
}
VIDEO_DIR = DATA_ROOT / "YouTubeClips"
OUTPUT_PKL = DATA_ROOT / "raw-captions.pkl"


def load_official_video_ids(split_files):

    all_ids = set()
    for split_name, file_path in split_files.items():
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.rsplit('_', 2)
                video_id = parts[0] if len(parts) == 3 else line
                all_ids.add(video_id)
    return all_ids

def find_video_file(video_id, video_dir):

    candidates = list(video_dir.glob(f"{video_id}*.avi"))
    if not candidates:
        return None
    candidates.sort(key=lambda x: len(x.stem))
    return candidates[0]


official_video_ids = load_official_video_ids(OFFICIAL_SPLITS)
print(f"官方划分中的总视频数: {len(official_video_ids)}")


df = pd.read_csv(CAPTION_CSV)
df = df[~df["Description"].isna()]
df = df[df["VideoID"].isin(official_video_ids)]


valid_video_ids = []
for vid in df["VideoID"].unique():
    video_file = find_video_file(vid, VIDEO_DIR)
    if video_file:
        valid_video_ids.append(vid)
    else:
        print(f"警告: 未找到视频 {vid}，相关标注将被过滤")
final_df = df[df["VideoID"].isin(valid_video_ids)]


def build_segment_captions(df, split_files):
    captions = {}
    for split_name, file_path in split_files.items():
        with open(file_path, "r") as f:
            for line in f:
                segment_id = line.strip()
                if not segment_id:
                    continue

                parts = segment_id.rsplit('_', 2)
                video_id = parts[0] if len(parts) == 3 else segment_id
                if video_id in df["VideoID"].values:
                    video_captions = df[df["VideoID"] == video_id]["Description"].tolist()
                    captions[segment_id] = [cap.strip().split() for cap in video_captions if isinstance(cap, str)]
                else:
                    print(f"警告: 片段 {segment_id} 的视频ID {video_id} 无对应标注")
    return captions

all_captions = build_segment_captions(final_df, OFFICIAL_SPLITS)


with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(all_captions, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"\n成功生成PKL文件: {OUTPUT_PKL}")
print(f"总片段数: {len(all_captions)}")
print(f"总描述数: {sum(len(v) for v in all_captions.values())}")