# RLCN-homework

本项目针对刚上手强化学习、Python、Pytorch编程的新任而设立，内容为RLCN2021作业的baselines以及相关资源汇总

## 平台地址
[作业提交](http://www.jidiai.cn/)

[课程地址](http://rlchina.org)

## 目录结构
```
|-- RLCN-homework
   |-- LICENSE
   |-- README.md
   |-- requirements.txt
   | -- test_main.py
   |-- course1
      | -- submission.py
      | -- test_main.py
   |-- course2
      | -- submission.py
      | -- q_table.pth
      | -- test_main.py
   |-- course3
      | -- submission.py
      | -- test_main.py
      | -- critic.py
      | -- critic.pth
   |-- course4
   |-- course5
  
```

## 环境搭建

- 安装最新版Anaconda
- 下载本项目中的requirements.txt文件
- 打开cmd，并进入到下载文件的目录
- pip install -r requirements.txt（会需要比较久的时间安装第三方包）
- 搭建完成

## 本地测试环境

本项目可以让用户本地测试算法，效果与平台相似，并且实时输出agent与环境的交互动画以及每个episode获得的reward

### 测试运行方法
- 将项目中的test_main.py文件复制到/SummerCourse2021/course?(?表示具体的课程号)/examples目录下
- 打开新的test_main.py，将submission.py的代码全部复制粘贴于#------submission-------#分割线间 
- 将其他在submission.py需要用到的文件全部复制到/SummerCourse2021/course?(?表示具体的课程号)/examples目录下
- 在该目录下打开命令行
- 将course?中README文件所提示的命令中的main.py改为test_main.py，并执行该命令
  
  例如：course3中应改为
  >python test_main.py --scenario classic_CartPole-v0 --algo dqn --reload_config 
- 本地测试完成

