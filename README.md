# HPCell

HPCell, a Hypergraph Transformer framework for predicting cell type abundance from spatial histology image.

![model.jpg](https://github.com/VitaIntelli-CQU/HPCell/blob/main/model.jpg)

## Installation  

We recommend using Anaconda to create a new Python environment and activate it via

```
conda create -n HPCell python=3.11
conda activate HPCell
```
Install the dependencies:
```
pip install -r requirements.txt
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
```

## Quick Start

### Input

* **train_data:**   Processed training data.
* **test_data:**    Processed test data.

### Output

* **pred:**   Cell abundance prediction.

### For calling Encoder programmatically

```python
tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=False, checkpoint_path=".../pytorch_model.bin")
```

Download the checkpoint on Huggingface.


## Training Models

We provide detailed training codes in `train.ipynb` to train with the example data we uploaded.

To prepare your own dataset, please follow `data_preparation_tutorial.ipynb`, users can train/finetune `HPCell` on their own dataset for further cellular analysis.


## Cellular Analysis and Evaluation

### Cell_abundance_visulization

- Use `HPCell` to visualize predicted fine-grained cell abundance

### Key_cell_evaluation

- Evaluate the prediction performance of `HPCell` on several key cell types

### Super_resovled_cell_abundance_visulization

- Produce super-resolved fine-grained cell type abundances using `HPCell` for biological reserach


This work uses parts of the code from [https://github.com/Weiqin-Zhao/Hist2Cell].

## Citation
If you find our codes useful, please consider citing our work:
```
@article{HPCell,
  title={Accurately predicting cell type abundance from spatial histology image through HPCell},
  author={Yongkang Zhao, YouYang Li, Weijiang Yu, Hongyu Zhang, Zheng Wang, Yuedong Yang, Yuansong Zeng},
  journal={},
  year={2025},
}
```
