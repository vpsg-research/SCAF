<div align="center">
<h1> [AAAI'26] Beyond Fully Supervised Pixel Annotations: Scribble-Driven Weakly-Supervised Framework for Image Manipulation Localization </h1>
</div>

## 📢 News
* **[2025-11]** Our paper is accepted by **AAAI 2026**! 🎉
* **[2025-11]** The code are being organized and will be released shortly. Please star this repo for updates!

## ✨ Highlights
* **⚠️First Scribble-Based IML Benchmark:** We construct **Sc-IML, the first scribble-annotated dataset for IML**. It effectively bridges the gap between expensive pixel-level annotations and coarse image-level labels, providing a solid foundation for the development and evaluation of weakly supervised IML methods and thereby advancing further research in this field.


* 🚀**First Scribble-Based Weakly Supervised IML Framework:** We propose a **scribble-based weakly supervised IML framework**, which integrates **structural consistency, prior-aware feature modulation, and gated adaptive fusion modules**, substantially improving model robustness and IML accuracy. Moreover, a **confidence-aware entropy minimization loss** regularizes predictions in weakly annotated and unlabeled regions, thereby further enhancing the model’s generalization.


* **🧩Boosting SOTA Performance:** The proposed model consistently **outperforms existing fully supervised methods** on both in-distribution and out-of-distribution test sets. Moreover, it remains highly robust and maintains accurate IML even under severe recompression and complex quality degradations on **real-world social platforms** (e.g., Facebook, Weibo, WeChat, WhatsApp).

##  Introduction
Official repository for the AAAI2026 paper *Beyond Fully Supervised Pixel Annotations: Scribble-Driven Weakly-Supervised Framework for Image Manipulation Localization* [[paper]](https://arxiv.org/abs/2507.13018) 

<div align="center">
    <img width="600" alt="image" src="Figures/introduction.png?raw=true">
</div>


Comparison between pixel-level masks and scribble annotations and their impact on IML performance. Top: Given an input image, a dense mask requires about 23 minutes to annotate, whereas a sparse scribble (blue for authentic, red for manipulated) captures the main structure of objects in only ~20 seconds, making scribbles roughly 69× faster than pixel-level annotations. Bottom: Radar plots show the  performance of different methods on in-distribution (left) and out-of-distribution (right) benchmarks. Our scribble-based weakly supervised IML model (red) consistently outperforms fully supervised baselines in terms of overall localization accuracy and robustness.

## 📻 Overview

<div align="center">
    <img width="1000" alt="image" src="figures/method.png?raw=true">
</div>


The overall architecture of the proposed SCAF. The model comprises two key modules: the prior-aware feature modulation module (PFMM) and the gated adaptive fusion module (GAFM). It is worth noting that the PFMM consists of a manipulated discriminator (MD) and a feature modulation module (FMM).

## 🎮 Getting Started

### 1. Install Environment

To set up the experimental environment, please follow the specific requirements for each baseline model. Taking **SCAF** as an example, you can create and install the environment using the provided script:

```bash
conda env create -f SCAF.yml
```
### 2. Prepare Datasets
| Dataset     | Nums        |  #CM          | #SP          | #IP          |  #Train          |  #Test          | 
| :----:      |    :----:   |         :----:|:----:        |    :----:    |         :----:   |         :----:  |
| CASIAv2   | 5123        | 3295          |1828          |    0         |        5123      |        0        |
| CASIAv1   | 920         | 459           |461           |    0         |        0         |        920      |
| Coverage    | 100         | 100           |0             |    0         |        70        |        30       |
| Columbia    | 180         | 0             |180           |    0         |         130      |        50       |
| NIST      | 564         | 68            |288           |    208       |        414       |        150      |
|CocoGlide| 512 | - |  - | - | 0  |  512  |
|ITW      |201  | 0 | 201|-  | 0  |  201  |
|Korus   | 220 | -|-|-|0|220|
|IMD|2010|-|-|-|0|2010|

- CASIAv2 [Download](https://github.com/SunnyHaze/IML-Dataset-Corrections)
- CASIAv1 [Download](https://github.com/SunnyHaze/IML-Dataset-Corrections)
- Columbia  [Download](https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/)
- Coverage  [Download](https://github.com/wenbihan/coverage?tab=readme-ov-file)
- NIST    [Download](https://mfc.nist.gov/users/sign_in)
- CocoGlide [Download](https://www.grip.unina.it/download/prog/TruFor/CocoGlide.zip)
- ITW [Download](https://minyoungg.github.io/selfconsistency/)
- Korus [Download](https://pkorus.pl/downloads/dataset-realistic-tampering)
- IMD [Download](https://staff.utia.cas.cz/novozada/db/)

We re-annotated several mainstream IML datasets, including 5,123 images from CASIAv2, 70 images from Coverage, 130 images from Columbia, and 414 images from NIST16, resulting in a total of 5,737 images. This constitutes the first scribble-based IML (**Sc-IML**) dataset [[Download](https://drive.google.com/file/d/1OoyTl6GV64TQnr-Mbnk5x9RdQ202MLve/view?usp=drive_link)]. In our annotations, "1" denotes foreground regions, "2" background regions, and "0" unlabeled areas. The image appears black when visualized because its pixel values lie in the range 0-255.

**Directory Structure:**
```
Dataset
├── train
│   ├── Image
│   ├── Scribble
│   ├── trainset_bg
│   └── trainset_fg
├── Test
|   ├── CASIAv1
|   |   |——test.txt
|   |   |——test
|   |   |  |——Image
|   |   |  |——GT
|   |   |  |——bg
|   |   |  └──fg
|   |—— Coverage
|   |   |——test.txt
|   |   |——test
|   |   |  |——Image
|   |   |  |——GT
|   |   |  |——bg
|   |   |  └──fg
....
....
|   |—— IMD
|   |   |——test.txt
|   |   |——test
|   |   |  |——Image
|   |   |  |——GT
|   |   |  |——bg
|   |   |  └──fg
```
All prior maps are available for download from [[Google Drive]](https://drive.google.com/file/d/1T18MU8Hno65nf-GDgg0kkb5oR1LFcQ3X/view?usp=drive_link).

### 3. Train

To train our SCAF model, first ensure that the pretrained [PVTv2-B2](https://github.com/whai362/PVT) weights are ready, and then simply run the corresponding training script.
```bash
# Make sure your directory paths are set correctly!
python etrain.py
```
### 4. Test
```bash
python etest.py
```

### 5. Evaluation
After confirming that the paths to your prediction and ground-truth files are correct, run the following script to compute pixel-level F1, IoU, and AUC.
```bash
python eval.py
```


## 🖼️ Experimental Results
We provide [training weights](https://drive.google.com/file/d/1H-0QRNDeTSgG69zP1fZibIOvP2LUPYST/view?usp=drive_link) presented in the papers.

<div align="center">
<img width="1000" alt="image" src="Figures/table.png?raw=true">
</div>

<div align="center">
Comparison with other  fully supervised IML methods. Bold and underlined indicate the best and second-best.
</div>

## ✨ Visualization results 

<div align="center">
    <img width="1000" alt="image" src="figures/vs.png?raw=true">
</div>

<div align="center">
Visualization results of different methods.
</div>

## Citation
If you find our code useful, please consider citing us and give us a star!


## 👍Credit
Our code is partly based on [CRNet](https://github.com/dddraxxx/Weakly-Supervised-Camouflaged-Object-Detection-with-Scribble-Annotations) and [PatchCore](https://github.com/amazon-science/patchcore-inspection.git). We sincerely thank their authors for their open-source contributions.