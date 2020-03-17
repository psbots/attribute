# Attribute
## Neural Network Interpretability and Visualization library for Tensorflow 2.0

This project is in its very early stages, so expect a lot of changes. It started as a port of the [saliency](http://github.com/pair-code/saliency) library to Tensorflow 2.0, and implementations of many other interpretability techniques are in the works (see list below).

## Installation

```
pip install git+git://github.com/psbots/attribute.git#egg=attribute
```

## Examples
![vanilla gradients](https://github.com/psbots/attribute/blob/master/res/vanilla.png)
![integrated gradients](https://github.com/psbots/attribute/blob/master/res/integrated.png)
[Notebook](https://github.com/psbots/attribute/blob/master/example.ipynb)


## Implementations :

- [x] Vanilla Gradients ([paper](https://scholar.google.com/scholar?q=Visualizing+higher-layer+features+of+a+deep+network&btnG=&hl=en&as_sdt=0%2C22),[paper](https://arxiv.org/abs/1312.6034))
- [x] SmoothGrad ([paper](https://arxiv.org/abs/1706.03825))
- [x] Integrated Gradients ([paper](https://arxiv.org/abs/1703.01365))
- [ ] Guided Backpropogation ([paper](https://arxiv.org/abs/1412.6806))
- [ ] Grad-CAM ([paper](https://arxiv.org/abs/1610.02391))
- [ ] XRAI ([paper](https://arxiv.org/abs/1906.02825))
- [ ] Occlusion
- [ ] Ablation-CAM ([paper](http://openaccess.thecvf.com/content_WACV_2020/papers/Desai_Ablation-CAM_Visual_Explanations_for_Deep_Convolutional_Network_via_Gradient-free_Localization_WACV_2020_paper.pdf))
- [ ] Full-Grad ([paper](https://arxiv.org/abs/1905.00780))
- [ ] Testing with Concept Activation Vectors (TCAV) ([paper](https://arxiv.org/abs/1711.11279) [code TF1.0](https://github.com/tensorflow/tcav))
- [ ] others...
