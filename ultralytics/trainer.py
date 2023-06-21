from ultralytics.yolo.data.dataset import YOLODataset
from ultralytics.yolo.utils.torch_utils import de_parallel
from ultralytics.yolo.utils import LOGGER, colorstr
from ultralytics.yolo.data.augment import Compose, Format, LetterBox, RandomFlip
from ultralytics import yolo  # noqa

def custom_transform(dataset, imgsz, hyp, stretch=False):
    """Convert images to a size suitable for YOLOv8 training."""

    flip_idx = dataset.data.get('flip_idx', None)  # for keypoints augmentation
    if dataset.use_keypoints:
        kpt_shape = dataset.data.get('kpt_shape', None)
        if flip_idx is None and hyp.fliplr > 0.0:
            hyp.fliplr = 0.0
            LOGGER.warning("WARNING ⚠️ No 'flip_idx' array defined in data.yaml, setting augmentation 'fliplr=0.0'")
        elif flip_idx and (len(flip_idx) != kpt_shape[0]):
            raise ValueError(f'data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}')

    transform = Compose([
        LetterBox(new_shape=(imgsz, imgsz)),
        RandomFlip(direction='vertical', p=hyp.flipud),
        RandomFlip(direction='horizontal', p=hyp.fliplr, flip_idx=flip_idx)])  # transforms
    
    return transform



class CustomDataset(YOLODataset):
  # TODO: use hyp config to set all these augmentations
  def build_transforms(self, hyp=None):
      """Builds and appends transforms to the list."""
      if self.augment:
          hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
          hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
          transforms = custom_transform(self, self.imgsz, hyp)
      else:
          transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
      transforms.append(
          Format(bbox_format='xywh',
                  normalize=True,
                  return_mask=self.use_segments,
                  return_keypoint=self.use_keypoints,
                  batch_idx=True,
                  mask_ratio=hyp.mask_ratio,
                  mask_overlap=hyp.overlap_mask))
      return transforms



def build_yolo_dataset(cfg, img_path, batch, data, mode='train', rect=False, stride=32):
    """Build YOLO Dataset"""
    return CustomDataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == 'train',  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == 'train' else 0.5,
        prefix=colorstr(f'{mode}: '),
        use_segments=cfg.task == 'segment',
        use_keypoints=cfg.task == 'pose',
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == 'train' else 1.0)

class CustomTrainer(yolo.v8.detect.DetectionTrainer):
    
    def build_dataset(self, img_path, mode='train', batch=None):
      """Build YOLO Dataset

      Args:
          img_path (str): Path to the folder containing images.
          mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
          batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
      """
      
      gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
      return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == 'val', stride=gs)

