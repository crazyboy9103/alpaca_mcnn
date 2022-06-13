# README
## Refer to options.py

## Setup environment
* Change prefix on the bottom of ```alpaca.yaml``` to conda path
* e.g. ```$PATH/anaconda3/envs/alpaca_tf```
* ```conda env create -f alpaca.yaml```
* ```conda activate alpaca_tf```

## Run main routine
* ```sh run_routine.sh```
    1. resize images to target_size and save them in proc_train_folder, proc_val_folder
    2. extract and resize gt density maps to gt_target_size and save them as train_pkl_file, valid_pkl_file
    3. Run training 

## Evaluate a model (mae, mape)
* Training will save best model ckpt in ```base_path/alpaca/YYYYMMDD-HHMMSS``` folder 
* Modify model_path to ```./alpaca/YYYYMMDD-HHMMSS```
* ```python evaluate.py```

## Rendering an image with saved model
* Modify model_path to ```./alpaca/YYYYMMDD-HHMMSS```
* ```python render_one_image.py```
