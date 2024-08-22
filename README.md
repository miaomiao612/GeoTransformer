# GeoTransformer
This is the official implementation of the paper "GeoTransformer: Enhancing Urban Forecasting with Geospatial Attention Mechanism". 
Please check the [paper] (https://arxiv.org/abs/2408.08852).

## Dependencies
First install pytorch associated with your cuda version, for example:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```
Then install the following packages:
```bash
pip install Pillow, lpips, diffusers, numpy, pandas, tqdm
```

## Datasets
We provide all the related data [here](https://huggingface.co/datasets/GeoTransformer/geotransformer/tree/main). Satelite images and census data are used for SAE model training and inference. GDP and Ride-share data are the prediction tasks. We also provide the inferenced urban latents and distances matrix for GeoTransformer model training and inferencing.

Below is the structure of the dataset directory:

<pre>
GeoTransformer/
│
├── datasets/                    
│   ├── ACS_results.json           # Census Data
│   ├── Trips_results.json         # Ride-share Demand
│   └── GDP_results.json           # GDP
│
├── distance_and_latents/ 
│   ├── center_distances.json      # distances between every two urban regions
│   └── latent_space.csv           # urban latent representations
│
└── rgb.zip                   
    ├── tile_0_71.tif              # satellite image for each urban region
    ├── ......
</pre>            



## SAE and Urban Latents (optional)
As we provide the urban latent representations data and the SAE training is very time consuming, you can skip this step and start training GeoTransformer model.
If you want to train SAE from scratch, first download the satellite images and put them under /rgb folder, then run the following command:
```bash
python SAE_train.py
```
The SAE model will be saved every 10 epochs, and then you can inference the urban latent using:
```bash
python SAE_inference.py --model_path SAE_epoch_100.pth
```


## GeoTransformer
To train GeoTransformer, first download the distances matrix data, urban latent representations data, GDP and Ride-share demand data and put them in the same folder of the training code, then run:
```bash
python GeoTransformer_train.py --train_type=gdp --num_heads=16 --num_layers=4 --knn=49 --l2_regularization=5e-4 --drop_out=0.05 --batch_size=4 --learning_rate =0.01 --epoch=60 --weighting_type=Linear --output_dir='./'
```
Each input variables can is explained below:
- **`--train_type`** (`str`): Choose the parameter to predict. Options are 'gdp' or 'trips'. Default is 'gdp'.
- **`--num_heads`** (`int`): Number of transformer attention heads. Default is 16.
- **`--num_layers`** (`int`): Number of transformer attention layers. Default is 4.
- **`--knn`** (`int`): Number of query neighbors. Default is 49.
- **`--l2_regularization`** (`float`): Weight decay factor for regularization. Default is 0.0005.
- **`--drop_out`** (`float`): Dropout rate for training. Default is 0.05.
- **`--batch_size`** (`float`): Training batch size. Default is 0.05.
- **`--learning_rate`** (`float`): Learning rate. Default is 0.05.
- **`--epoch`** (`float`): Number of training epochs. Default is 0.05.
- **`--weighting_type`** (`float`): Type of the transformer weighting type, choose from 'Linear', 'IDW','Gaussian'. Default is 0.05.
- **`--output_dir`** (`float`): The output directory of the models and the the report.

## Evaluation
The training code of GeoTransformer will print out the MSE, MAE and R-squared on both train/test set at each training epoch. The result of all training steps will also be saved as a report.txt file.

## Baselines
We include codes for baseline models. You can modify the setting inside their codes.