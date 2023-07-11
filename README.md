# DeepFloyd IF (K8S 集群自用)

原项目地址：https://github.com/deep-floyd/IF
## 使用 IF model 的显存需求:

<p align="center">
  <img src="./pics/deepfloyd_if_scheme.jpg" width="100%">
</p>

- 目前下载好的模型有：`IF-II-L-v1.0`, `IF-I-M-v1.0`, `IF-I-XL-v1.0`, `t5-v1_1-xxl`, `CLIP`
- 存储地址位于：`/mnt/nfs/share/deepfloydIF`
-  `IF-I-M-v1.0`, `IF-I-XL-v1.0` 是文字转图片模型，`IF-II-L-v1.0`是超分辨模型。可以根据模型大小估算相应显存需求，编写 yaml 文件。模型大小参见上图。

## 使用说明：
- 由于模型以及相关配置文件都已经下载到本地，所以原则上不会发生网络通信
- 主文件入口为 `dream.py` 和 `img2img.py`
    - `dream.py` 是文字转图片的 pipeline
    - `img2img.py` 是基于输入图片生成符合 text_embedding 以及 style_embedding 语义的图片
    - pipeline 保留了生成过程中的中间计算结果，储存在 img 文件夹内
- 由于 torch.float16 没有内置实现 cos_vml_cpu 等 cpu 运算（具体细节请参考[飞书内部文档](https://aicarrier.feishu.cn/wiki/XRLmw1h2ziiR4ckZxupcd11Ynmd#JYoCd6q8OocEUwx6n9DcAdFRntb)），所以如果想将 IF 模型加载进 cpu 或 gpu，需要配置正确的 config 和 pipeline 参数
    - 以 `IF-I-M-v1.0` 为例，若想加载至 cpu ，需要如下改动
        - 将 `/mnt/nfs/share/deepfloydIF/weights/IF_/IF-I-M-v1.0/config.yml` 中的 `precision` 改为 "32"
        - 将 pipeline 中的 device 改为 'cpu'
    - 以 `IF-I-M-v1.0` 为例，若想加载至 gpu ，需要如下改动
        - 将 `/mnt/nfs/share/deepfloydIF/weights/IF_/IF-I-M-v1.0/config.yml` 中的 `precision` 改为 "16"
        - 将 pipeline 中的 device 改为 'cuda:0'

## 运行命令：
```shell
pip install -r requirements.txt
# 文字->图片
python dream.py
# 图片+文字->图片
python img2img.py
```