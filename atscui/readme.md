# 交通信号智能体训练图形界面

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

```
env = createEnv(config)
model = createAgent(env, config).model

model_obj = Path(config.model_path)
if model_obj.exists():
    print("load model=====加载训练模型==在原来基础上训练")
    model.load(model_obj) 

obs = ... # 来自探测器的状态变量
action, _ = model.predict(obs)  # 将状态变量转换为动作变量

# 将动作action（改变灯态的指令串）发送给信号机

```