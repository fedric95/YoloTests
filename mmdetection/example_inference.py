from mmdet.apis import DetInferencer

call_args = {
    'inputs'         : 'mmdetection/demo/demo.jpg',
    'out_dir'        : 'outputs2/',
    'pred_score_thr' : 0.3,
    'batch_size'     : 1
}

init_kw = {
    'model'          : 'rtmdet_tiny_8xb32-300e_coco.py',
    'weights'        : 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth',
    'device'         : 'cpu',
    'palette'        : 'none'
}

inferencer = DetInferencer(**init_kw)
inferencer(**call_args)
