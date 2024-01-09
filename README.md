# *CQS-Attention* documentation

## Overview
*CQS-Attention* is a sequence parallelism technique for standard attention computation. 

The underlying model is simply fork-join. It consists of three components: Scheduler, Workers, Tiler. Scheduler equally partitions computation responsibility in a **mutually exclusive** manner and ensures the tokens involved in local computation is **minimum**. Each worker **independently** computes the standard attention of the assigned subsequence and transfers local results to Tiler, which organizes all local results and produces the final attention. 

The greatest advantage of *CQS-Attention* is the low memory requirement for a single device: suppose it requires $\mathbb{X}$ memory to execute the standard attention computation on a single device, with *CQS-Attention*, each of $W$ worker devices only needs $\frac{1}{W}\mathbb{X}$ memory size as each will execute the standard attention computation of a subsequence, which contains approximately $\frac{1}{\sqrt{W}}$ of all tokens.  

Since *CQS-Attention* is a fork-join model, one very important by-product is the speedup advantage. For simplicity, we ignore the data transfer cost and define the speedup as the ratio of the computation time of the whole sequence to that of the longest subsequence, thus $\mathcal{S} = \frac{t_{\text{whole sequence}}}{t_{\text{longest subsequence}}}$. We define the performance curve of a worker device as the curve that describes its computation time with the number of tokens. Therefore, the speedup of *CQS-Attention* depends on the performance curve of the worker device in deployment. In the paper, we employ NVIDIA A100 GPU. Detailed results and analysis can be found in the paper.

*CQS-Attention* is also special because the responsibility partition of worker devices is completely mutually exclusive. Being free of communication among workers avoids many design complexities such as synchronization, race condition, etc. Moreover, mutual exclusion introduces more potentials. For example, communication between Scheduler-Workers and Workers-Tiler can be asynchronous; If $W$ devices can compute the attention of a long sequence, one device can do the same computation, but in $W$ time units. etc. More discussions can be found in the paper.

In this repo, we provide demo code to show the workflow of *CQS-Attention* (a.k.a. case study 1 in the paper). The demo code also proves the correctness of computation, provides memory consumption summaries, and visualizes the partition of the $N \times N$ matrix.

## Contents of Repository

```
SPASL_v1 repository
├── README.md (this file)
├── src
│   ├── MODULE_CQS-Attention.py
│   ├── MODULE_utils.py
│   └── MODULE_visualize.py
├── Interest_Sets
│   ├── 3
│   │   └── 3.txt
│   ├── ...
│   └── 111
│       └── 111.txt
├── demo
│   ├── demo.ipynb
│   └── demo-output.pdf
└── CaseStudy2
    ├── Active_Memory_Timeline_Result
    │   ├── Usage.txt
    │   ├── W_1.pickle
    │   ├── ...
    │   └── W_31.pickle
    ├── active_memory_timeline_demo.ipynb
    ├── case_study_2_demo.ipynb
    └── case-study-2-demo-output.pdf
```
### 1. src
It contains the source code for this repo.

### 2. Interest_Sets
It contains interest set(s) for $W=3\dots 111$. They are determined using exhaustive search by previous researchers.

Specifically, for $W<53$, all interest sets are provided in ```#_full.txt```, except for $W=43$ (There are too many). We randomly select $20$ of all interest sets, if more than $20$, and include them in ```#.txt```.

For $W\ge53$, only the first interest set was provided by previous researchers, and we did not bother searching for more because when the sequence length ($N$) is large, the different among interest sets is very trivial.

### 3. demo
Readers are recommended to run ```demo.ipynb``` for a better understanding of how *CQS-Attention* works. Configuring the Python environment is trivial. In this notebook, setting $N=10,W=7,d=1,\mathcal{I}=[0,1,3]$, case study 1 in the paper can be reproduced. Readers are encouraged to try other configurations. Note that, the priority of *CQS-Attention* implementation in this repo readability, at the sacrifice of efficiency and speed. For example, all cell locations are explicitly generated but it is not necessary because the mapping is consecutive: only need to record the start and end index, etc.

One output example of the demo code is provided in ```demo-output.pdf```.

### 4. CaseStudy2
The folder ```Active_Memory_Timeline_Result``` contains memory timeline files of case study 2 described in Appendix F in the paper.

```active_memory_timeline_demo.ipynb``` is the code for the generation of GPU active memory timeline files.

Readers can easily reproduce case study 2 using ```case_study_2_demo.ipynb```, very similar wall-clock time numbers should be observed on NVIDIA A100 GPU (80GiB).

Again, we provide the output of ```case_study_2_demo.ipynb``` in ```case-study-2-demo-output.pdf``` for readers to inspect without running the code.

