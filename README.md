```
docker run -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $PWD:/yolo/ nvcr.io/nvidia/pytorch:23.03-py3 /bin/bash
```


python YOLOX/tools/demo.py image -f YOLOX/exps/default/yolox_s.py -c yolox_s.pth --path YOLOX/assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu

python YOLOX/tools/train.py -f YOLOX/exps/example/custom/yolox_s.py -d 0 -b 64 --fp16 -o -c yolox_s.pth


python -m yolox.tools.train -n yolox-nano -d 1 -b 64 --fp16



python YOLOX/tools/train.py -f exp.py -d 0 -b 64 --fp16 -o -c yolox_s.pth



