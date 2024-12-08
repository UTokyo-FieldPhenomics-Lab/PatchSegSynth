# PatchSegSynth: A Patch-Level Data Synthesis Pipeline Enhances Species-level Crop and Weed Segmentation in Natural Agricultural Scenes

![Pipeline Diagram](figs/grabs.png)
Proposed patch-level data synthesis pipeline.
Our pipeline can improve species-level segmentation of crops and weeds in agricultural scenes by up to **15%**.

## Dataset
This project makes use of the `WE3DS dataset` (Weed, Crop, and Soil Dataset) for training and evaluation purposes. The WE3DS dataset provides annotated RGB-D images for semantic segmentation in agricultural scenes.
**Reference**:  
Kitzler, F.; Barta, N.; Neugschwandtner, R.W.; Gronauer, A.; Motsch, V.  
**WE3DS: An RGB-D Image Dataset for Semantic Segmentation in Agriculture**.  
*Sensors* **2023**, 23, 2713. [https://doi.org/10.3390/s23052713](https://doi.org/10.3390/s23052713)

## To utilize our pipeline on the `WE3DS dataset`

1. Download the [`WE3DS.zip`](https://zenodo.org/records/7457983) and unzip the `WE3DS/` to the root of this repo.

2. Configure the `src/configuration.py`. Please specify `WE3DS_PATH` to the `WE3DS` folder and the ouput path `SYNTHETIC_PATH` (by default, results will be output to `/Synthetics`)

3. Split train and test set by running `src/001_split_train_test_folder.py`
    ```
    python 001_split_train_test_folder.py
    ```

4. Extract individual patches (crops, weeds and soils) to form foreground and background pool by running `src/002_crop_patches.py`
    ```
    python 002_crop_patches.py
    ```
5. Generate synthetic sample by running `src/003_pathc_level_data_synthesis.py`. Run the script with the `-x` option to specify how many times the base dataset size (1540) should be generated. The generated images and masks will be saved in the output directory named according to the specified multiplier.
    ```
    python 003_patch_level_data_synthesis.py -x <multiplier>
    ```

## Evaluation
### Datasets Evaluated
This project compares the performance of plant species semantic segmentation models using:

1. **WE3DS Dataset** (real data)  
2. **Synthetic Datasets** (generated synthetic data)  
3. **Hybrid Datasets** (real + synthetic)

### Metrics

- **IoU (Intersection over Union)**: Evaluates segmentation accuracy per class.  
- **mIoU**: Mean IoU across all classes.  
- **mIoU (No-Soil)**: Excludes the soil class for better plant species assessment.  

### Performance and Training Time for Hybrid Datasets at Different Ratios

The hybrid datasets was evaluated to identify the optimal combination of real and synthetic data for plant species semantic segmentation.

| **Hybrid Ratio**            | **mIoU** | **mIoU (No-Soil)** | **Training Time (hours)** |
|------------------------------|----------|--------------------|---------------------------|
| Original Baseline Dataset   | 0.646    | 0.626              | 3.04                      |
| 1:1 Hybrid                  | 0.683    | 0.664              | 5.78                      |
| 1:5 Hybrid                  | 0.722    | 0.706              | 16.63                     |
| 1:10 Hybrid                 | 0.732    | 0.716              | 29.87                     |
| 1:15 Hybrid                 | **0.734**| **0.719**          | 39.88                     |
| 1:20 Hybrid                 | 0.720    | 0.704              | 57.12                     |

Our results demonstrated that increasing data volume substantially improves performance, achieving a maximum
mIoU increase of `15%` at 15x, though gains diminish beyond 15×. Hybrid datasets achieved optimal performance at a `10×` ratio, balancing accuracy and efficiency.