from mmseg.datasets import CustomDataset, DATASETS

@DATASETS.register_module()
class MyDataset(CustomDataset):
    # ✅ 旧版 MMSegmentation 要求显式定义 CLASSES 和 PALETTE
    CLASSES = ('background', 'object1', 'object2')
    PALETTE = [[0, 0, 0], [128, 128, 128], [255, 255, 255]]

    # ✅ 新版结构（可保留 METAINFO，不冲突）
    METAINFO = dict(
        classes=CLASSES,
        palette=PALETTE,
    )
