import os
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "tfl_data")
OUTPUT_PATH = os.path.join(BASE_DIR, "processed_data", "merged_tfl_data.csv")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


DAYFIRST_FILES = {"401", "402", "403", "404"}


df_list = []
print("开始合并 TfL 数据文件（按不同日期格式解析）...\n")

all_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])

for filename in all_files:
    file_path = os.path.join(DATA_DIR, filename)
    file_prefix = filename[:3]

    try:
        try:
            df = pd.read_csv(file_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="ISO-8859-1")

       
        df.columns = df.columns.str.strip().str.replace("\n", " ").str.replace('"', '')

        if "Start date" not in df.columns:
            print(f"跳过：{filename} 缺少 'Start date'")
            continue

        
        if file_prefix in DAYFIRST_FILES:
            df["Start date"] = pd.to_datetime(df["Start date"], dayfirst=True, errors="coerce")
            format_flag = "dayfirst=True"
        else:
            df["Start date"] = pd.to_datetime(df["Start date"], errors="coerce")
            format_flag = "dayfirst=False"

        valid = (~df["Start date"].isna()).sum()
        print(f"{filename}: {len(df)} 行，有效 Start date: {valid} 行（{format_flag}），范围: {df['Start date'].min()} ~ {df['Start date'].max()}")

        df_list.append(df)
    except Exception as e:
        print(f"读取失败: {filename} 错误: {e}")

# 合并并保存
if df_list:
    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n 合并完成，已保存至：{OUTPUT_PATH}")
    print(f"总记录数：{len(merged_df)}")

    merged_df["month"] = merged_df["Start date"].dt.to_period("M")
    print("\n 每月记录数：")
    print(merged_df.groupby("month").size())
else:
    print("没有成功合并任何文件，请检查文件格式。")
