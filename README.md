# EfficientPlace: Reinforcement Learning within Tree Search for Fast Macro Placement

This is the code for our paper "Reinforcement Learning within Tree Search for Fast Macro Placement". Zijie Geng, Jie Wang, Ziyan Liu, Siyuan Xu, Zhentao Tang, Mingxuan Yuan, Jianye Hao, Yongdong Zhang, Feng Wu. ICML 2024.

## Environment
To build the environment, you can follow commands in `scripts/environment.sh`.

## Usage
To run the code,
```
python main.py benchmark@_global_=adaptec1
```

## Citation
If you find this code useful, please consider citing the following paper.
```
@inproceedings{
    geng2024reinforcement,
    title={Reinforcement Learning within Tree Search for Fast Macro Placement},
    author={Zijie Geng and Jie Wang and Ziyan Liu and Siyuan Xu and Zhentao Tang and Mingxuan Yuan and Jianye HAO and Yongdong Zhang and Feng Wu},
    booktitle={Forty-first International Conference on Machine Learning},
    year={2024},
    url={https://openreview.net/forum?id=AJGwSx0RUV}
}
````

Hyperparameters setting:
````
num of update epochs: 10
update frontiers freq: 2
update frontiers begin: 200
num of episodes: 5
density weight in (phase2): 0
number of iterations (phase2): 1000
density weight in (phase3)
````

Dataset setting: ISPD2005 benchmark (training from scratch)
````
Adaptec group: bao gồm adaptec1, adaptec2, adaptec3, adaptec4.
Bigblue group: bao gồm bigblue1, bigblue2, bigblue3, bigblue4.

Note: Bigblue2 và bigblue4 have total macros too big, Authors setted up by only use 256 and 1024 macro to for placement, to justify for comparing balance
````


DRL Checkpoint could be found in this url: https://drive.google.com/drive/folders/1uUzJk4j_a0nuAfciroKjserOduwTV1tG?usp=drive_link
Contact me if there was some problem about searching checkpoint: nhatranvan204@gmail.com
