# Refer to options.py

# Run main routine
* sh run_routine.sh
    1. resize images to target_size and save them in proc_train_folder, proc_val_folder
    2. extract and resize gt density maps to gt_target_size and save them as train_pkl_file, valid_pkl_file
    3. Run training 

# Evaluate a model (mae, mape)
* Modify model_path 
* python evaluate.py

# Rendering an image with saved model
* Modify model_path
* python render_one_image.py
