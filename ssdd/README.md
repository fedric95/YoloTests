docker run -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $PWD:/yolo/ nvcr.io/nvidia/pytorch:23.03-py3 /bin/bash

cd /yolo/
git clone https://github.com/ultralytics/ultralytics && cd ultralytics && pip install -e . && cd ../ssdd/

