# *CQS_Attention* documentation

## Overview
*CQS_Attention* is a sequence parallelism technique for standard attention computation. 

The underlying model is simply a fork-join one. It consists of three components: Scheduler, Workers, Tiler. Scheduler equally partitions computation responsibility in a **mutually exclusive** manner and ensures the tokens involved in local computation is **minimum**. Each worker **independently** computes the standard attention of the assigned subsequence and transfers local results to Tiler, which organizes all local results and produces the final attention. 

The greatest advantage of *CQS_Attention* is the low memory requirement for a single device: suppose it requires $\mathbb{X}$ memory to execute the standard attention computation on a single device, with *CQS_Attention*, each of $W$ worker devices only needs $\frac{1}{W}\mathbb{X}$ memory size as each will execute the standard attention computation of a subsequence, which contains approximately $\frac{1}{\sqrt{W}}$ of all tokens.  

Since *CQS_Attention* is a fork-join model, one very important by-product is the speedup advantage. For simplicity, we ignore the data transfer cost and define the speedup as the ratio of the computation time of the whole sequence to that of the longest subsequence, thus $\mathcal{S} = \frac{t_{\text{whole sequence}}}{t_{\text{longest subsequence}}}$. We define the performance curve of a worker device as the curve that describes its computation time with the number of tokens. Therefore, the speedup of *CQS_Attention* depends on the performance curve of the worker device in deployment. In the paper, we employ NVIDIA A100 GPU. Detailed results and analysis can be found in the paper.

*CQS_Attention* is also special because the responsibility partition of worker devices is completely mutually exclusive. Being free of communication among workers avoids many design complexity such as synchronization, race condition, etc. Moreover, mutual exclusion introduces more potentials. For example, communication between Scheduler-Workers and Workers-Tiler can be asynchronous; If $W$ devices can compute the attention of a long sequence, one device can do the same computation, but in $W$ time units. etc. More discussions can be found in the paper.

In this repo, we provide demo code to show the workflow of *CQS_Attention* (a.k.a. case study 1 in the paper). The demo code also proves the correctness of computation, provides memory consumption summaries, and visuzlizes the partition of the N by N matrix.

## Contents of Repository

```
SPASL_v1 repository
├── README.md (this file)
├── demo.ipynb
├── demo_requirement.txt
├── demo_output.pdf
├── dataset
│   ├── SPASL_v1 (SH, PI, AA, SN, LR: 1000 classes)
│   └── SPASL_v1_portable
│       ├── SPASL_v1 (AA: 1000 classes)
│       └── img_to_gen (1000 classes)
├── demo
│   ├── SPASL_v1_demo
│   └── result_demo
├── full_result
│   ├── IW_table.zip
│   ├── Radar_chart.zip
│   ├── Ranking.md
│   ├── SPAPS_v1_full_test_results.xlsx
│   └── SPASL_v1_full_test_results.json
└── src
    ├── step1_test_on_ImageNet_training_set
    ├── step2_SPASL_generation
    ├── step3_test_on_SPASL_v1
    ├── utils
    ├── visualize
    └── requirements.txt
```
### 1. demo.ipynb, demo_output.pdf, demo_requirement.txt
In ```demo.ipynb```, we guide the reader thorugh this work using four experiments. They demonstrate ideas of curve fitting, the distribution of $\lambda$ values, cross search and usage of a SPASL benchmark.

In case some readers decide to skip running the demo code, we provide the outputs in ```demo_output.pdf```. It can be used for fast output check, comparisons, etc.

We encourage the reader to run demo in a Python virtual environment. We provide a brief tutorial in ```demo.ipynb``` and all the required packages are listed in ```demo_requirement.txt```.

### 2. dataset
```
dataset
├── SPASL_v1 (6 GB)
└── SPASL_v1_portable (1.7 GB)
    ├── SPASL_v1 (AA: 1000 classes)
    └── img_to_gen (1000 classes)
```
The whole set will not be available until the publication of the work. It will be uploaded to a repo, which may reveal the author information.

We provide two ways of obtaining SPASL\_v1: download as a whole piece (6 GB), or download a portable version (1.7 GB) if the user has already downloaded ImageNet-1K. In SPASL\_v1\_portable, we provide the full AA component and the list of source images to extract from ImageNet-1K for the rest four components. Becasue AA example generation is time-consuming compared to the other components. 

In SPASL_v1, there are $48\,442$ distinct source images from ImageNet-1K. 
We do not provide all source images of four components so that the user can apply the modification on is becasue there are $39\,343$ of them. The difference from $40\,000$ is insignificant and the portable version will have the same size. 


### 3. demo
```
demo
├── SPASL_demo (54 MB)
│   ├── SH (14 MB, 10 classes)
│   ├── PI (1.6 MB, 10 classes)
│   ├── AA (17 MB, 10 classes)
│   ├── SN (21 MB, 10 classes)
│   └── LR (0.4 MB, 10 classes)
└── exp_materials
    ├── AlexNet_n01440764_after_AA.csv
    ├── IW_table_general_correct_idx_n01440764.csv
    ├── IW_table_general_lambda_n01440764.csv
    ├── IW_table_general_summary_n01440764.csv
    ├── IW_table_vit_lambda_n01440764.csv
    └── MaxVit_n01440764_after_AA.csv
```

For the demonstration purpose, we provide a tiny subset of SPASL\_v1 as SPASL\_demo. It contains $10$ classes, thus, $10\times10\times5=500$ images in total. The size of this demo dataset is $54$ MB, the size of each component is listed above.

After testing a model on SPASL\_demo, e.g. resnet_18, related results will be generated and placed in the ```demo/result``` folder as shown below.
```
result
├── IW_table
│   └── resnet_18
│       ├── n01440764
│       │   ├── resnet_18_IW_table_SH_n01440764.csv
│       │   └── ... (5 IW tables in total)
│       └── ... (10 classes in total)
└── result_on_whole_SPASL
    ├── history.csv (an entry is added)
    ├── history.json (an entry is added)
    └── resnet_18
        ├── resnet_18_top1_acc.csv
        ├── resnet_18_top5_acc.csv
        └── resnet_18_radar_chart_on_SPASL_demo.png
```
The generated results are explained as follows.

>**{model_name}\_IW\_table\_{component_name}\_{class_name}.csv**

It provides the image-wise prediction details of {model_name} on all $10$ images in {class_name} in {component_name}, with the $\lambda$ value calculated. Specifically, for each image, $(Prob_0, Prob_1, Prob_2, Prob_3, Prob_4, correct\_idx, \lambda)$ is listed.

>**{model_name}\_top1/5\_acc.csv**

It provides the test results of {model_name} on SPASL\_demo, thus, top-1 and top-5 accuracies on each component, and SPASL score.

>**{model_name}\_radar_chart\_on\_SPASL\_demo.png**

It is the plot of top-1 and top-5 accuracies in radar chart.

>**history.csv/.json**

It is the local test history on SPASL\_demo. 

### 4. full_result
```
full_result
├── IW_table.zip
├── Ranking.md
├── SPASL_v1_full_test_radar_chart.zip
├── SPAPS_v1_full_test_results.xlsx
└── SPASL_v1_full_test_results.json
```

The ranking of $80$ very influential models on SPASL_v1 is provided in ```Ranking.md```.

The last three files are the detailed test results of these models.

### 5. src
```
src
├── step1_test_on_ImageNet_training_set
├── step2_SPASL_generation
├── step3_test_on_SPASL_v1
├── utils
├── visualize
└── requirements.txt
```

This folder provides all the source code for benchmark generation, model testing, test result visualization, etc. The required packages are listed in ```requirements.txt```
