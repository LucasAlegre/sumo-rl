# 交通信号智能体训练图形界面

本模块是原ui模块的重构。将原模块拆分为3个子模块：界面子模块，配置子模块，算法子模块。

## 模块化设计

- （1）界面组件子模块
  - 配置界面
  - 绘图界面
- （2）配置项子模块
  - 基础配置
  - 算法私有配置
- （3）算法模型子模块
  - 常用4种算法：DQN, PPO, A2C, SAC
- （4）辅助函数

## 操作类型(operation)

- TRAIN: 根据配置训练智能体，并保存模型(model.zip)和日志(conn*.csv)
- EVAL: 评估模型，保存评估结果(eval.txt)
- PREDICT: 用模型预测，保存预测结果(predict.json)

## 运行命令

在项目根目录执行命令：``` python atscui/main.py ```

## 加载模型进行预测

run_model.py是一个简易的命令行程序，指定net.xml,rou.xml,model_path,algo_name,
加载训练后的模型，根据状态选择动作，让环境执行该动作。

测试结果符合预期，即相同的状态observation，会产生相同的动作action。

```
python atscui/run_model.py 
```

```
config = parse_config()
env = createEnv(config)
model = createAlgorithm(env, config.algo_name)
model_obj = Path(config.model_path)

if model_obj.exists():
    print("==========load model==========")
    model.load(model_obj)

obs = env.reset()
for _ in range(10):
    action, _ = model.predict(obs)  # 通过状态变量预测动作变量
    # 将动作action（改变灯态的指令串）发送给信号机
    print(obs)
    for _ in range(3):
        obs, reward, done, info = env.step(action)  # 来自探测器的状态变量
        print(action)
```