# LMH 2025-3-28
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class ImmuneCell_Dataset(BaseSegDataset):
    # 类别和对应的 RGB配色
    METAINFO = {
        'classes': ['background', 'anthracotic material', 'engulfed anthracotic material', 'lymphocyte', 'suspected lymphocyte', 'neutrophil', 'suspected neutrophil', 'eosinophil', 'suspected eosinophil'],
    'palette': [
    [0, 0, 0],          # 背景 - 纯黑
    [255, 0, 0],        # 炭末物质 - 纯红
    [255, 165, 0],      # 吞噬炭末 - 橙色
    [0, 100, 0],        # 淋巴细胞 - 深绿
    [144, 238, 144],    # 疑似淋巴细胞 - 浅绿
    [0, 0, 255],        # 中性粒细胞 - 纯蓝
    [173, 216, 230],    # 疑似中性粒细胞 - 浅蓝
    [128, 0, 128],      # 嗜酸性粒细胞 - 紫色
    [255, 192, 203]     # 疑似嗜酸性粒细胞 - 粉红
]
    }
    
    # 指定图像扩展名、标注扩展名
    def __init__(self,
                 seg_map_suffix='.png',   # 标注mask图像的格式
                 reduce_zero_label=False, # 类别ID为0的类别是否需要除去
                 **kwargs) -> None:
        super().__init__(
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
