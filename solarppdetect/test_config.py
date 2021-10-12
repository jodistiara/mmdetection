from mmcv import Config
from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
import mmcv

# the new config inherits the base configs to highlight the necessary modification
cfg = Config.fromfile('solarppdetect/config/cascade_mask_rcnn_swin-s-p4-w7_fpn_1x_coco_2.py')
cfg.work_dir = './solarppdetect'

# Build dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

print(f'Config:\n{cfg.pretty_text}')
print("=========\nModel")
print(model)
print("=========\nModel's CLASSES")
print(model.CLASSES)