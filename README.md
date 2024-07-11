TAKT: Target-Aware Knowledge Transfer for Whole Slide Image Classification (MICCAI-2024)
===========
[Conghao Xiong](https://bearcleverproud.github.io/), [Yi Lin](https://ianyilin.github.io), [Hao Chen](https://cse.hkust.edu.hk/~jhc/), [Hao Zheng](https://scholar.google.com.hk/citations?user=LsJVCSoAAAAJ&hl=zh-CN), [Dong Wei](https://scholar.google.com.hk/citations?hl=zh-CN&user=njMpTPwAAAAJ), [Yefeng Zheng](https://sites.google.com/site/yefengzheng/), [Joseph J. Y. Sung](https://www.ntu.edu.sg/about-us/leadership-organisation/profiles/professor-joseph-sung) and [Irwin King](https://www.cse.cuhk.edu.hk/irwin.king/home)

ArXiv | MICCAI

<img src="framework.jpg" width="1000px" align="center" />

**Abstract:** Knowledge transfer from a source to a target domain is vital for whole slide image classification, given the limited dataset size due to high annotation costs. However, domain shift and task discrepancy between datasets can impede this process. To address these issues, we propose a Target-Aware Knowledge Transfer framework using a teacher-student paradigm, enabling a teacher model to learn common knowledge from both domains by actively incorporating unlabelled target images into the teacher model training. The teacher bag features are subsequently adapted to supervise the student model training on the target domain. Despite incorporating the target features during training, the teacher model tends to neglect them under inherent domain shift and task discrepancy. To alleviate this, we introduce a target-aware feature alignment module to establish a transferable latent relationship between the source and target features by solving an optimal transport problem. Experimental results show that models employing knowledge transfer outperform those trained from scratch, and our method achieves state-of-the-art performance among other knowledge transfer methods on various datasets, including TCGA-RCC, TCGA-NSCLC, and Camelyon16.

## Updates:
* 2024 July 11th: Created this repository and first push. The code is still under organization, so please stay tuned if you are interestd!
