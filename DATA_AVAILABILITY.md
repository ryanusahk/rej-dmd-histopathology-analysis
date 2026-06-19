# Data Availability And Expected Inputs

Full re-execution of these notebooks and scripts requires large microscopy data and intermediate analysis artifacts that are not included in this code deposit.

Expected local inputs include, but may not be limited to:

- `serial.csv`
- `scan_metadata_master.xls`
- `cropped/`
- `cropped_cells/`
- Cellpose mask files and segmentation pickles
- Cellpose custom model files, such as files under `custom_models/`
- `quantified_cells.csv`
- `20250308_merged_df.csv`
- `202050404_abe_df.csv`
- `202050404_minidystrophin_df.csv`
- `minimyo_tri_df.csv`
- `active_learning_output/`
- Active-learning label CSVs and model pickle files
- `power_analysis_cache/`

Large raw image files, derived segmentation objects, trained models, and cached simulation outputs should be stored separately from the code repository when file size or data-sharing restrictions apply.

The code is provided to document the computational workflow used for manuscript analysis. Users attempting full re-execution should recreate the expected directory structure or update paths inside the notebooks/scripts to match their local data locations.

