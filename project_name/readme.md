## python搜索路径

1. 修改 PYTHONPATH 环境变量
一种方法是通过设置 PYTHONPATH 环境变量，将项目根目录（sumo-rl）添加到 Python 的模块搜索路径中。

2. 使用相对导入（推荐）
如果你希望在 project_tests 中直接运行测试文件，而不依赖于修改 PYTHONPATH，你可以通过相对导入来解决。

```
from ..project_name import module1, module2, config, utils

cd /Users/xnpeng/sumoptis/sumo-rl
python -m project_tests.test_module1 
```

3. 修改 sys.path（临时解决方法）
如果你不想通过设置环境变量或相对导入，也可以在 test_module1.py 中直接修改 sys.path，让 Python 能找到项目根目录。

```
os.path.dirname(__file__) 获取当前文件的路径。
os.path.join(..., '../') 获取父目录路径。
os.path.abspath() 将路径转换为绝对路径。
sys.path.insert(0, ...) 将绝对路径添加到 Python 的模块搜索路径列表中。
```