## 说明

tune是ray的超参数微调框架。就是在指定的范围内尝试不同的参数值进行训练，再比较训练效果。

## 基本概念 tune_concepts.py

- tune_trail()
- num_samples()
- search_space()
- tune_scheduler()
- bayesopt()
- tune_trainable_function()
- tune_trainable_class()

## 编程指南 tune_start.py

```
tuner = tune.Tuner(
    train_mnist,
    param_space=search_space,
)
results = tuner.fit() 
```

## 完整例子 tune_hyperopt_example.py

## 基础例子 tune_basic_example.py

## 在checkpoint上微调 save_load_checkpoint.py

