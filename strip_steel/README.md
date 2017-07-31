# 使用说明

## 预处理
预处理脚本将原始瑕疵图片数据进行数据集切分，形成训练集和测试集（默认80:20），并截取为$100\times100$的图片。

```
python preprocessing.py /path/to/image_directory
```
其他选项：

* --prefix，输出文件前缀，默认为output
* --test-split，测试集比例，默认为20%

执行预处理后，程序会在当前目录生成4个文件，分别是

* output-test.txt，测试集图片路径
* output-train.txt，训练集图片路径
* output-data.npz，预处理后的训练集数据
* classes.txt，瑕疵类别

## 训练
进入`cnn`目录内，执行

```
python cnn_train.py /path/to/train_dataset.npz
```

默认使用1E-4的初始学习率，并进行指数衰减，终端会输出训练进程
![train](media/train.png)

训练过程中会输出checkpoint文件，默认放在`cnn/train`目录内。

## 训练重启
重启是一种提高训练模型的技巧：模型训练过程中学习率在指数下降，到达饱和后，可以将学习率重新提高到初始值重新下降，这样错误率一开始可能会上升，但随后会到达更低的程度。

```
python cnn_train.py /path/to/train_dataset.npz --load_ckpt --ckpt_step xxx(step number)
```
![error rate](media/error_rate.png)
<p align="center">错误率曲线</p>

经过两次重启，错误率最终降低到2E-3以下。

## 测试
由于测试集图片大小不一，而且与训练模型输入不同，我们采取一个简单的策略，在测试图片中随机截取N张$64\times64$（与训练模型保持一致）的图片进行测试，并将N个预测结果合并得到测试图片的类别。

```
python cnn_eval.py --data_file ../output-test.txt --ckpt_file train/model.ckpt-xxx --class_def ../classes.txt
```

终端会输出每张图片预测的结果，并统计时间。
![eval](media/eval.png)
<p align="center">测试结果</p>