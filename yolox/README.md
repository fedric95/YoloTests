```
pip install 'YOLOX @ git+https://github.com/Megvii-BaseDetection/YOLOX.git'
```
```
python -m yolox.tools.train -f yolox_exp_ssdd.py -d 2 -b 32 --fp16 -o -c yolox_s.pth
```