# Automated Exploration of Optimal Neural Network Structures for Deepfake Detection

![license](https://img.shields.io/badge/License-MIT-brightgreen)
![python](https://img.shields.io/badge/Python-3.8-blue)
![pytorch](https://img.shields.io/badge/PyTorch-1.13.1+cu116-orange)


## Requirements

- Python 3.8.5
- PyTorch 1.13.1


## Usage

### Preparation

1. Prepare dataset Celeb-DF ([reference](https://github.com/yuezunli/celeb-deepfakeforensics)).

2. Place dataset in a directory like below.

```
/data
│
├── Celeb-real-image-face-90
│   ├── id25_0006_016.jpg # Preprocessed real image
│   ├── id25_0006_017.jpg # Preprocessed real image
│   ├── id25_0006_018.jpg # Preprocessed real image
│   └── ...
│
└── Celeb-synthesis-image-face-90
    ├── id28_id26_0003_225.jpg # Preprocessed fake image
    ├── id28_id26_0003_226.jpg # Preprocessed fake image
    ├── id28_id26_0003_227.jpg # Preprocessed fake image
    └── ...
```

3. Create file `/.env` as below (Please specify the directory where the image is placed).

```
FAKE_DATA_PATH=/data
```



### DARTS

1. Run the command below.

`python ./darts/cnn/train_search_celeb.py --unrolled`


### PC-DARTS

1. Run the command below.

`python ./pc-darts/train_search_celeb.py --unrolled`


### DU-DARTS

1. Run the command below.

`python ./du-darts/train_search_celeb.py --unrolled`


## Citation

blank


## License

MIT License


## Acknowledgement

We extend our sincere gratitude to the [DARTS](https://github.com/quark0/darts), [PC-DARTS](https://github.com/yuhuixu1993/PC-DARTS), and [DU-DARTS](https://github.com/ShunLu91/DU-DARTS) projects for their foundational contributions to this work. Our code significantly benefits from their innovations, and we are deeply thankful for the opportunity to build upon their remarkable efforts.
