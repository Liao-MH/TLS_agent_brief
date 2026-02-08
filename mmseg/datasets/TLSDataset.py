# LMH 2025-10-5
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class TLSDataset(BaseSegDataset):
    # 类别和对应的 RGB配色
    METAINFO = {
        # 'classes':['background', 'TLS', 'GC', 'Immune Cluster'],
        # 'palette':[[127,127,127], [0,0,100], [50,50,50], [20,40,60] ] # RGB格式
        'classes':['background', 'TLS', 'GC'],
        'palette':[[127,127,127], [0,0,100], [50,50,50]] # RGB格式
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
