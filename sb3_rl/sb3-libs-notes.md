## conda create -n sumoai-sb3 python=3.10

## libs:

sb3-libs-ver.txt

### 模型算法的训练、保存、加载、评估、预测，以及训练的视频

#### 1，训练 model.learn(...)

#### 2，保存 model.save(...)

#### 3，加载 model.load(...)

#### 4，评估 model.evaluate(...)

#### 5，预测 model.predict(...)

#### 6，训练的视频

```
env = VecVideoRecorder(...)
(1) env
(2) video_folder
(3) record_video_trigger
(4) video_length
(5) name_prefix
(6) metadata
```

## atari_game_a2c : 

```
conda create -n sumoai-atari python=3.10
pip install setuptools==57.1.0
pip install stable-baselines3

```

运行 ```python atari_game_a2c.py``` 时，抛出异常：```Environment `PongNoFrameskip` doesn't exist.``` 尚无法解决。

