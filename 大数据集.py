from datasets import load_dataset, concatenate_datasets
import pandas as pd

target_langs = ["ar", "bg", "zh", "de", "el", "en", "es", "fr",
                "hi", "ru", "sw", "th", "tr", "ur", "vi"]

samples_per_lang = 5000
dataset_name = "MoritzLaurer/multilingual-NLI-26lang-2mil7"

# 1. 先加载 dataset 的 splits 信息（不带 split 参数会返回 DatasetDict，包含所有 splits）
print("Loading dataset metadata / available splits ...")
ds_dict = load_dataset(dataset_name)  # 这会返回一个 DatasetDict，键是可用的 splits 名称
available_splits = list(ds_dict.keys())
print(f"Found {len(available_splits)} splits. Example splits: {available_splits[:12]}")

# 2. 对每种语言，收集以该语言代码开头的 split 名称
lang_to_splits = {}
for lang in target_langs:
    pref = f"{lang}_"
    # 有些 split 可能直接是语言名开头（如 'en_mnli'），也可能有其他命名，使用 startswith 匹配
    matches = [s for s in available_splits if s.startswith(pref)]
    # 有极少数数据集 split 可能命名为仅 language code（如果存在也包含）
    if lang in available_splits:
        matches.append(lang)
    lang_to_splits[lang] = matches
    print(f"{lang}: {len(matches)} splits")

# 3. 逐语言加载对应 splits 并合并，然后抽样 samples_per_lang 条
all_lang_frames = []
for lang in tqdm(target_langs, desc="Languages"):
    splits = lang_to_splits.get(lang, [])
    if not splits:
        print(f"Warning: no splits found for language {lang}, skipping.")
        continue

    # 逐 split 加载并追加到列表（使用 ds_dict 已经加载的部分以避免重复 download）
    lang_datasets = []
    for s in splits:
        # ds_dict[s] 已经是一个 Dataset 对象（从最开始 load_dataset(dataset_name) 得到）
        ds_split = ds_dict[s]
        lang_datasets.append(ds_split)

    # 合并该语言所有 split
    if len(lang_datasets) == 1:
        lang_ds = lang_datasets[0]
    else:
        lang_ds = concatenate_datasets(lang_datasets)

    # 随机打乱并选取前 samples_per_lang 条
    # 注意：shuffle() 会返回新的 Dataset（随机 seed 保证可复现）
    lang_ds_shuffled = lang_ds.shuffle(seed=42)
    take_n = min(samples_per_lang, len(lang_ds_shuffled))
    if take_n == 0:
        print(f"Warning: {lang} has 0 examples after merging splits.")
        continue
    lang_sampled = lang_ds_shuffled.select(range(take_n))

    # 转为 pandas 并追加
    df_lang = pd.DataFrame(lang_sampled)
    # 确保包含必要列（premise, hypothesis, label, language）
    expected_cols = ["premise", "hypothesis", "label", "language"]
    have_cols = [c for c in expected_cols if c in df_lang.columns]
    df_lang = df_lang[have_cols]
    df_lang["source_lang"] = lang  # 标注来源语言（可选）
    all_lang_frames.append(df_lang)

    print(f"Sampled {len(df_lang)} examples for language {lang}")

# 4. 合并所有语言 DataFrame
if all_lang_frames:
    df_all = pd.concat(all_lang_frames, ignore_index=True)
    # 可选：全局打乱
    df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\nTotal combined samples: {len(df_all)}")
else:
    df_all = pd.DataFrame()
    print("No samples collected.")

df_x = df_all