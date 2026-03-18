# IMMARL: Interpretable Multi-Modal Attentive Representation Learning for Accurate RNA Property Prediction

For any questions or further information, please reach out to Haitao Fu (Email: fuhaitao@hubu.edu.cn, https://ai.hubu.edu.cn/info/1025/2041.htm).

## Project Structure

`dataset_multimodal.py`: Handles multi-modal data loading (1D sequence, 2D topology, 3D geometry), PDB parsing for RNA structures, graph construction, and dataset-specific cleaning logic (Covid, Tcribo, Fungal).

`dataset.py`: Provides foundational RNA encoding utilities, including sequence numerical mapping and secondary structure pairing logic .

`loader.py`: Manages data partitioning and batch loading logic, supporting both random splits and fixed train/val/test splits for specific RNA datasets.

`models/`: Directory containing specific neural network encoders for each data modality:

- **`transformers.py`**: Implementation of the Transformer encoder with self-attention for 1D sequence feature extraction.
- **`gnns.py`**: GCN (Graph Convolutional Network) implementation for capturing RNA secondary structure as 2D graphs.
- **`egnn.py`**: Equivariant Graph Neural Network (EGNN) for processing 3D atomic coordinates and pooling them into nucleotide-level representations.

`model_multimodal.py`: Contains the core end-to-end fusion model, including:

- **Aligned-Cross-Attention**: Strategic fusion of multi-modal features via cross-modal layers.
- **Auxiliary Predictor**: Multi-task learning component designed to force the EGNN to learn effective 3D representations.

`run_multimodal.py`: Main entry point for the training pipeline, covering optimization (Adam), weighted MCRMSE loss calculation, validation with early stopping, and performance testing.

`run_ablation.py`: Specialized script for ablation studies to evaluate the individual contribution of 1D, 2D, and 3D modalities.

`analysis_viz.py`: Visualization module for generating paper-quality scatter plots and attention heatmaps to interpret model decisions.

`utils.py`: Utility functions for structural processing (PDB alignment, bond computation) and core evaluation metrics (PCC, Spearman, $R^2$).

## Dataset

All datasets used in this research are openly available from these sources:

- COVID：https://www.kaggle.com/competitions/stanford-covid-vaccine
- Fungal：https://academic.oup.com/mbe/article/39/2/msab372/6513383
- Tc-Riboswitches：https://pubs.acs.org/doi/10.1021/acssynbio.8b00207

## Requirements  

torch==2.5.0  
NumPy==1.26.4  
torch_geometric==2.6.1  
SciPy==1.14.1  
scikit-learn==1.5.2  
pandas==2.2.2  
tqdm==4.66.5  
networkx==3.3   
rdkit==2024.03.5  
Optuna==4.4.0 

### Run Main Experiments:  

Take running  on CovidVaccine for example:

`nohup python run_multimodal.py --dataset covid --device cuda:0 --epochs 300 --batch_size 16 --lr 0.0005 --d_model_1d 128 --dim_ff_1d 256 --hidden_2d 128 --hidden_3d 128 --num_layers_1d 4 --L_2d 4 > covid_train.log 2>&1 &`  
