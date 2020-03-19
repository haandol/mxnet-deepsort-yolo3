# Introduction

Example of MXNet YOLO3 based Deep-SORT implementation.

In order to generate mis detected images for finetune YOLO3.

Deep-SORT implementation is mostly based on [nwojke/deep_sort](https://github.com/nwojke/deep_sort).

# Quick Start

1. Install dependencies

```
$ pip install -r requirements.txt
```

2. Copy your own video file into the folder to test

3. Run demo with BYO-video

```bash
$ python demo.py --src video.mp4 --out_dir images --fps 15
```

4. Output Images. Frame images will stored into output directory along with mis-detected objects.

```bash
$ ls images
```