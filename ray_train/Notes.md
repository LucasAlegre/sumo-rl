## Ray train

- save and load checkpoint: config正确
- save and load checkpoint: restore不正确。

不能解决checkpoint restore的问题，实验无法继续下去。

## 上面问题的解决办法
很简单，恢复训练时是接着上次训练的回合(epoch)继续训练的，因此在配置里面，必须将训练回合数设置为超过上次的回合数。
```
train_loop_config={"num_epochs": 10} 
```
