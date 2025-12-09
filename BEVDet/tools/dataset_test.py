# test_data_read.py
# 简单粗暴的数据读取测试脚本（硬编码版）
# 用途：验证 cfg.data.train 是否能成功 build & 迭代一个 batch

import os
import traceback
from mmcv import Config
import torch
import debugpy
from mmdet3d.datasets import build_dataset, build_dataloader

# ====== 按需修改这两个硬编码变量 ======
CONFIG_FILE = '/inspire/hdd/project/wuliqifa/chenxinyan-240108120066/songbur/goodet/BEVDet/configs/bevdet/bevdet-r50-cbgs.py'  # ← 你的配置文件
SAMPLES_PER_GPU = 1
WORKERS_PER_GPU = 0
# ===================================

def summarize(name, obj, indent=0, max_items=6):
    """尽量简洁地打印内容结构/shape，方便定位问题"""
    pad = '  ' * indent
    try:
        if isinstance(obj, torch.Tensor):
            print(f"{pad}- {name}: Tensor, dtype={obj.dtype}, shape={tuple(obj.shape)}")
        elif isinstance(obj, (list, tuple)):
            print(f"{pad}- {name}: {type(obj).__name__}, len={len(obj)}")
            for i, v in enumerate(obj[:max_items]):
                summarize(f"{name}[{i}]", v, indent+1)
            if len(obj) > max_items:
                print(f"{pad}  ... ({len(obj)-max_items} more)")
        elif isinstance(obj, dict):
            print(f"{pad}- {name}: dict, keys={list(obj.keys())[:max_items]}")
            for k in list(obj.keys())[:max_items]:
                summarize(k, obj[k], indent+1)
            if len(obj.keys()) > max_items:
                print(f"{pad}  ... ({len(obj.keys())-max_items} more)")
        else:
            print(f"{pad}- {name}: {type(obj).__name__} -> {str(obj)[:120]}")
    except Exception as e:
        print(f"{pad}- {name}: <print error> {e}")

def main():
    print("==== 1) 加载配置 ====")
    assert os.path.exists(CONFIG_FILE), f"配置文件不存在: {CONFIG_FILE}"
    cfg = Config.fromfile(CONFIG_FILE)
    print("cfg 加载成功。")

    debugpy.listen(("0.0.0.0", 9876))
    print("[debugpy] listening on, waiting for VS Code to attach...")
    debugpy.wait_for_client()        
    print("attached")

    # 为了可重复性（可选）
    cfg.seed = 0

    print("\n==== 2) 构建训练集 ====")
    try:
        dataset = build_dataset(cfg.data.train)
        print(f"训练集类型: {type(dataset).__name__}, 样本数: {len(dataset)}")
    except Exception:
        print("构建训练集失败！下面是完整报错：")
        traceback.print_exc()
        return

    print("\n==== 3) 打印数据流水线（前几步） ====")
    try:
        # dataset.pipeline 是 Compose([...])，直接打印即可
        print(dataset.pipeline)
    except Exception:
        print("无法打印 pipeline（可能不是标准 Compose），跳过。")

    print("\n==== 4) 构建 DataLoader 并取一个 batch ====")
    try:
        loader = build_dataloader(
            dataset,
            samples_per_gpu=SAMPLES_PER_GPU,
            workers_per_gpu=WORKERS_PER_GPU,
            dist=False,
            shuffle=False
        )
        data_batch = next(iter(loader))
        print("成功取到一个 batch。下面打印关键字段与 shape：\n")
        if isinstance(data_batch, dict):
            for k, v in data_batch.items():
                summarize(k, v)
        else:
            summarize("data_batch", data_batch)

    except Exception:
        print("DataLoader 或取 batch 失败！下面是完整报错：")
        traceback.print_exc()
        return

    print("\n==== 5) 简单检查常见关键键（存在即 OK）====")
    expected_keys = [
        'points', 'gt_bboxes_3d', 'gt_labels_3d', 'img_inputs', 'img_metas'
    ]
    if isinstance(data_batch, dict):
        for k in expected_keys:
            print(f"- {k}: {'OK' if k in data_batch else '缺失'}")
    else:
        print("data_batch 不是 dict，跳过键检查。")

    print("\n✅ 数据读取测试完成。若上方没有报错，说明基本数据读取正常。")

if __name__ == "__main__":
    main()
