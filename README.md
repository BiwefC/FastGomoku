# FastGomoku

使用 AlphaZero 论文中的方法实现了一个五子棋的程序。

平台：

* Ubuntu
* cmake
* python3
* pyTorch

使用流程：

1. 在 `c` 文件夹下新建 `build` 文件夹并通过 cmake build 工程。
2. 进入 `python` 文件夹`python3 init.py`
3. `./run.sh`

简要说明：

* `run.sh`中对python脚本进行了调用；
* `config.py`为配置文件可以根据需要修改，lr需要根据训练速度修改，进程数可以根据自己的电脑配置；

训练速度：

i5 8400, GTX950, 3线程跑的情况下，两天后我已经下不过它了....

上述情况，断断续续跑了一周，20000步模拟状态下能执黑击败Yixin2014....
