### Data
- here we have the 2d slices for AC PET and NAC PET
    - "/mnt/data/mij17663/nac2ac_sep/2d_slices"
####
- did train/test split
    - used for this:
        - /home/mij17663/code/petct-img-translation/dataset_prep/split_data.py
    - following studies have been moved to test set (10% of studies, splitted by patients)
        - ['0219', '0089', '0129', '0050', '0058', '0192', '0144', '0178', '0132', '0174', '0116', '0068', '0105', '0224', '0071', '0025', '0222', '0072', '0202', '0020', '0251', '0008', '0195', '0040', '0054', '0233', '0084']
    - together we have 10469 slices in Test and 94383 in Tr

### ToDo
1. split data based on patients into training and validation set
    - done
2. write dataset
    - done
3. write dataloader
    - done
4. I also updated gan_model.py
    - in the current version discriminator takes 256x256 as input
    - later I want to rather changed it to patching
5. also updated train.py
    - it is ready for testing
    - we save a few images after each epoch so we can see the quality of generated images
    - we also check some evaluation metrics on the validation dataset