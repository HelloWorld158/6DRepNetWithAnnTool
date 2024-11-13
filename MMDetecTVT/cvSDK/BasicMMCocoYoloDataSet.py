import os
import BasicUseFunc as basFunc
from mmyolo.registry import DATASETS
from BasicMMCocoDataSet import EhlXMLCocoDataSet,EhlJsonCocoDataSet
from mmyolo.datasets.yolov5_coco import BatchShapePolicyDataset
@DATASETS.register_module()
class EhlXMLYoloCocoDataSet(BatchShapePolicyDataset,EhlXMLCocoDataSet):
    pass
@DATASETS.register_module()
class EhlJsonYoloCocoDataSet(BatchShapePolicyDataset,EhlJsonCocoDataSet):
    pass