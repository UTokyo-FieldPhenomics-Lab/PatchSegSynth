

### Path Configurations ###

# directory to WE3DS folder
WE3DS_PATH = '../WE3DS'

WE3DS_train_txt = f'{WE3DS_PATH}/train.txt'
WE3DS_test_txt = f'{WE3DS_PATH}/test.txt'
WE3DS_images = f'{WE3DS_PATH}/images'
WE3DS_annotations = f'{WE3DS_PATH}/annotations/segmentation/SegmentationLabel'


# target directory for data synthesis
SYNTHETIC_PATH = '../Synthetics'

# Real images folder (Real Baseline Train)
real_train_images = f'{SYNTHETIC_PATH}/Train/images'
real_train_annotations = f'{SYNTHETIC_PATH}/Train/annotations'
real_test_images = f'{SYNTHETIC_PATH}/Test/images'
real_test_annotations = f'{SYNTHETIC_PATH}/Test/annotations'

# Folder to save the individual patches
synthetic_fore_crop = f'{SYNTHETIC_PATH}/patch_pools/fore/crop'
synthetic_fore_weed = f'{SYNTHETIC_PATH}/patch_pools/fore/weed'
synthetic_back_soil = f'{SYNTHETIC_PATH}/patch_pools/back/soil'

######## WE3DS DATASET ########
# Soil
soil = ["soil"]

# Crop Species (Seven)
crop_species = [
    "broad_bean",
    "corn",
    "pea",
    "cornflower",
    "soybean",
    "sunflower",
    "sugar_beet"
]

# Weed Species (Ten)
weed_species = [
    "corn_spurry",
    "red-root_amaranth",
    "common_buckwheat",
    "red_fingergrass",
    "common_wild_oat",
    "corn_cockle",
    "milk_thistle",
    "rye_brome",
    "narrow-leaved_plantain",
    "small-flower_geranium"
]
# Class Names and Colors
class_names = [
    'void',
    'soil',
    'broad_bean',
    'corn_spurry',
    'red-root_amaranth',
    'common_buckwheat',
    'pea',
    'red_fingergrass',
    'common_wild_oat',
    'cornflower',
    'corn_cockle',
    'corn',
    'milk_thistle',
    'rye_brome',
    'soybean',
    'sunflower',
    'narrow-leaved_plantain',
    'small-flower_geranium',
    'sugar_beet'
]

class_colors = [
    '255,255,255',
    '0,0,0',
    '0,128,0',
    '128,128,0',
    '0,0,128',
    '128,0,128',
    '0,128,128',
    '128,128,128',
    '64,0,0',
    '192,0,0',
    '64,128,0',
    '192,128,0',
    '64,0,128',
    '192,0,128',
    '64,128,128',
    '192,128,128',
    '0,64,0',
    '128,64,0',
    '0,192,0'
]