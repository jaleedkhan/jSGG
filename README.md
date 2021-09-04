# Scene Graph Benchmark in Pytorch

## (Updated) Scene Graph Benchmark
Hello! This custom repository simple builds on the excellent work of Kaihua Tang (https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch). This repository was created as an easy way to run this particular scene graph generator in a notebook environment. To run the scene graph generator, you can simply download the Python Notebook file titled 'j_SGG.ipynb' (https://github.com/jaleedkhan/jSGG/blob/main/j_SGG.ipynb), upload it to Google Colab, and run it to completion. The internal code has been altered quite a lot to cater to these customized needs. Specifically, new dataloaders were defined after significant changes, so that the implementation of the function 'im2scenegraph' would be possible.

The function 'im2scenegraph' provides a convenient way of using the scene graph generator on custom images. The function can be passed any custom image, and will produce a complete scene graph, along with visualizations within the same notebook. I hope it can provide a good starting point for people looking for an instant way to run the SGG repository, without all the hassle of the setup.

This repository has been built and tested on Python 3 (PyTorch 1.6 Python 3.6 GPU Optimized) kernel on a ml.g4dn.xlarge instance on Amazon Sagemaker, and also on Google Colab.

Base paper: Tang et al. "Unbiased Scene Graph Generation from Biased Training", CVPR 2020. (https://arxiv.org/abs/2002.11949)

Main repository: Scene Graph Benchmark in Pytorch (https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)
