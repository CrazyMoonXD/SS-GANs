### SS-GANs:Text-to-Image via Stage by Stage Generative Adversarial Networks (Published in PRCV2019)
We propose a novel structure of network named SS-GANs, in which specific modules are added in different stages to satisfy the unique requirements. In addition, we also explore an effective training way named coordinated train and a simple negative sample selection mechanism. <br>
![](https://github.com/CrazyMoonXD/SS-GANs/blob/master/overall_structure.jpg)
### Dependencies
Python 2.7<br>
PyTorch >= 0.4.0
### How to train flowers/birds model:
```cd code/```<br>
```python main.py --cfg cfg/birds_3stages.yml --gpu 0```
### How to evaluate(inception scores):
```cd evaluate/```<br>
Flowers-102:<br>
```python flowers.py --image_folder path of your images```<br>
CUB-birds-200:<br>
```python birds.py --image_folder path of your images```
### Pretrained models:
64-flowers:<br>
[https://pan.baidu.com/s/1T5bdHRPdzuhTeApF44a9MQ](https://pan.baidu.com/s/1T5bdHRPdzuhTeApF44a9MQ) 提取码：5dnj<br>
128-flowers:<br>
[https://pan.baidu.com/s/1JQThOMX2cOC_Kgimoeqy5w](https://pan.baidu.com/s/1JQThOMX2cOC_Kgimoeqy5w) 提取码：c1y0<br>






