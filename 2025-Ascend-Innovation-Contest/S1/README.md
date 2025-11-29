# 2025年昇腾AI创新大赛-昇思模型开发挑战赛（S1赛季）
## 基本信息
比赛链接：[昇腾AI创新大赛-昇思模型开发挑战赛（S1赛季）](https://www.hiascend.com/developer/contests/details/21ffd6733ab54dc4b6b686a242c5d586?module=8fe62319958941e89a5b7b922fb01928)

获奖名单：[获奖名单](https://mp.weixin.qq.com/s/ITHLEqEXew03yHcmdt6cSg)

## 作品归档指南
详见WIKI：[Issue与PR提交规范](https://github.com/mindspore-courses/competition/wiki/%E6%8F%90%E4%BA%A4%E8%AF%B4%E6%98%8E)

请获奖选手在作品在[MoE](./MoE/)或[MultiModal](./MultiModal/)目录下创建以**队名**为目录名的子目录，将作品代码归档到该子目录下。同时提交一个README.md文件，说明你的优化内容、性能收益及其他相关细节. 最终目录结构为:
```
├── MoE
│   ├── 队伍A
│   │   ├── patches.zip
│   │   ├── README.md
├── MultiModal
│   ├── 队伍B
│   │   ├── patches.zip
│   │   ├── README.md
```
## 学习指南
如果需要学习获奖选手的作品, 可参考如下步骤:
```bash
unzip patches.zip # 解压你要学习的patch
git clone -b 0.4 https://github.com/mindspore-lab/mindnlp.git
cd mindnlp
git apply ../patches/*
```


## 代码工程归档
归档于[task1](./task1/)目录下。
## 赛题开发指导手册归档
该手册对应比赛页面的[赛题详情](https://www.hiascend.com/developer/contests/details/21ffd6733ab54dc4b6b686a242c5d586?module=e3e5672b8ae44bb4b2c1bdfed236a702)中的指导手册, 为防止链接失效在此归档。
### 赛题详情
选手任选一类模型，在精度无损下提升推理性能，基于动态图进行性能优化，模型类别如下：
1. MoE类：
- deepseek-ai/deepseek-moe-16b-chat
- Qwen/Qwen1.5-MoE-A2.7B-Chat
2. 多模态类：
- deepseek-ai/Janus-Pro-7B
- Qwen/Qwen2-VL-2B-Instruct

要求：
1. 优化前后不得损失精度（使用text或logits差异进行衡量）
2. 基于MindSpore框架进行开发
3. 获奖作品在昇腾社区、昇思社区等社区开源

最终将根据同一大类下多个模型的Prefill延时、Decode延时、峰值显存占用三方面的优化率进行加权排序。
### 环境准备

本赛题使用华为云modelarts-Notebook，开发环境，镜像为：`mindspore_2.6.0rc1-cann_8.1.rc1-py_3.10-euler_2.10.11-aarch64-snt9b`，硬盘规格推荐使用150G，实例规格`Ascend: 1*ascend-snt9b1|ARM: 24核 192GB`（`deepseek-ai/deepseek-moe-16b-chat`需要使用`Ascend: 1*ascend-snt9b3|ARM: 24核 192GB`）

启动notebook后，打开终端执行：
```bash
wget  https://mindspore-contest.obs.cn-southwest-2.myhuaweicloud.com/task1.zip
unzip task1.zip  -d task1
cd task1
bash init.sh
```
**task1已备份于[task1](./task1/)目录下。**

init.sh脚本中mindnlp仓库克隆仅需创建notebook时执行一次，以后每次重启不需要重新克隆。
### 模型权重准备 


终端中执行：
```bash
bash download.sh
```

下载模型，可根据选择的赛题按需下载。

### 启动推理


终端中执行：
```bash
bash test.sh
```
测试模型。最终结果将保存到eval_output/目录下。
### 作品调试    
选手可各显神通完成推理加速，可使用但不限于以下方法：
- 自定义算子接入: 自定义算子|MindSpore
- JIT: mindspore.jit|昇思MindSpore社区
- 异构计算


禁止使用以下方法：
- 使用pytorch等非MindSpore框架进行局部或全局替换实现；
- 不保证精度的情况下进行的计算裁剪；


选手可先基于`Ascend: 1*ascend-snt9b1|ARM: 24核 192GB`进行调试，**最终后台判分以`Ascend: 1*ascend-snt9b3|ARM: 24核 192GB`上性能为准**。

### 作品提交

所有的优化均应基于mindnlp仓库，选手需基于0.4分支将代码变动使用以下方式打包成   `patch.zip`

```bash
cd mindnlp #切换到mindnlp所在目录
git format-patch –${n} -o patches # 其中变量n为你commit到本地仓库的若干提交。
zip -r patches.zip patches # 打包成zip文件
```
最终提交打包的patches.zip文件。

### 自动判题
后台程序会将选手提交的patch文件应用到mindnlp仓库，然后启动判题脚本（前述的test.sh文件），比较选手提交后的模型输出与目标输出的logtis是否一致，若不一致则会在昇腾社区前端显示-1，若一致则会在昇腾社区显示对应的各项指标参数。

判题程序使用的环境如下：
- 镜像：`mindspore_2.6.0rc1-cann_8.1.rc1-py_3.10-euler_2.10.11-aarch64-snt9b`
- 实例规格：`Ascend: 1*ascend-snt9b3|ARM: 24核 192GB`
- 其他依赖版本：与`task1/init.sh`中一致


选手可在相同环境下进行自验，启动推理后，会在eval_output下给出结果。

**请注意：**
1. 为保证判题程序每次运行资源一致，判题程序每次仅会执行一个团队的提交，若提交人数过多需排队执行。
2. 判题程序执行结果会有合理范围内的波动。
3. 参赛者可多次提交，以最后提交为准。
### 参考链接
1. [MindSpore官网](https://www.mindspore.cn/)
2. [MindSpore代码仓](https://gitee.com/mindspore/mindspore)
3. [MindSpore NLP仓库](https://github.com/mindspore-lab/mindnlp)
4. [JIT推理优化教程（第五章节）](https://www.hiascend.com/developer/courses/detail/1925362775376744449)
5. [JIT优化文档](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/mindspore/mindspore.jit.html)
6. [自定义算子接入](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0/custom_program/op_custom.html)
7. [模型优化参考案例](https://www.mindspore.cn/news/detail?id=3770&type=technology-blogs)

