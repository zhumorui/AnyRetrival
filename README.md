# AnyRetrival: Enhancing Zero-shot Image Retrieval with Vision Foundation Models


This repository evaluates the performance of DINOv2 models on image retrieval tasks using the ROxford5k and RParis6k datasets. DINOv2 models leverage self-supervised learning to extract robust and generalizable features without relying on labeled data. The results show that DINOv2 achieves state-of-the-art performance in various retrieval scenarios, particularly excelling in high-quality and moderate-difficulty tasks. By supporting flexible input resolutions (224x224 and 448x448) and scalable model sizes (from vits14 to vitg14), DINOv2 adapts effectively to diverse use cases, offering a balance between computational efficiency and retrieval accuracy. Notably, DINOv2 surpasses traditional supervised models like DELG in easy scenarios while demonstrating competitive performance in hard retrieval challenges.


We would like to thank the authors of the [revisitop repository](https://github.com/filipradenovic/revisitop) for providing a helpful evaluation script and tools for image retrieval benchmarks. These resources have made it easier to reproduce results and compare different retrieval models.
