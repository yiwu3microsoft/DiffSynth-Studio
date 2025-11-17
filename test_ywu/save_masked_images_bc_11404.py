import os, json, pdb
from PIL import Image
import argparse
# { original/backgroud_change_training: 26G, 69G, 697M
#     "record_id": "0a0f1c4c1b4a37f1997b5bf364d3d9db",
#     "original_image_url": "https://localcreative.blob.core.windows.net/test/backgroud_change_training/art_collecting_data_30k_3category_ori_seg/0a0f1c4c1b4a37f1997b5bf364d3d9db.jpg",
#     "segmentation_image_url": "https://localcreative.blob.core.windows.net/test/backgroud_change_training/art_collecting_data_30k_3category_ori_seg/0a0f1c4c1b4a37f1997b5bf364d3d9db_seg.png",
#     "original_image_path": "/home/yiwu3/projects/data/data_br1u42-s2-09/data_background_change_11404/original/backgroud_change_training/art_collecting_data_30k_3category_ori_seg/0a0f1c4c1b4a37f1997b5bf364d3d9db.jpg",
#     "segmentation_image_path": "/home/yiwu3/projects/data/data_br1u42-s2-09/data_background_change_11404/segmentation/backgroud_change_training/art_collecting_data_30k_3category_ori_seg/0a0f1c4c1b4a37f1997b5bf364d3d9db_seg.png",
#     "segmentation_mask_path": "/home/yiwu3/projects/data/data_br1u42-s2-09/data_background_change_11404/segmentation_mask/backgroud_change_training/art_collecting_data_30k_3category_ori_seg/0a0f1c4c1b4a37f1997b5bf364d3d9db_seg.png",
#     "product_info": "Black corset with belt.",
#     "image_description": "Black corset with belt is placed on a rustic scene, with weathered bricks surrounding it, creating an adventurous atmosphere."
# }


parser = argparse.ArgumentParser(description="Example script using args")

# Add arguments
parser.add_argument("--data_root", type=str, required=True, help="Path to input file")


args = parser.parse_args()
data_root = args.data_root

with open(os.path.join(data_root, "metadata_bc_11404_diffsynth_format.json"), 'r', encoding='utf-8') as f:
    info_all = json.load(f)

os.makedirs(os.path.join(data_root, "masked_images"), exist_ok=True)

# pdb.set_trace()
count = 0
for info in info_all:
    count += 1
    file_name = info['image'].split('/')[-1]
    name = file_name.split('.')[0]
    print(f"Processing {count}/{len(info_all)}: {info['image']}")

    img = Image.open(os.path.join(data_root, info['image']))
    
    fname_mask = os.path.join(data_root, info['image'].replace("original", "segmentation_mask").replace(file_name, f"{name}.seg.png"))
    if not os.path.exists(fname_mask):
        fname_mask = os.path.join(data_root, info['image'].replace("original", "segmentation_mask").replace(file_name, f"{name}_seg.png"))
        if not os.path.exists(fname_mask):
            print(f"Mask not found for {info['image']}, skipping.")
            continue
    mask = Image.open(fname_mask).convert("L")

    masked_img = Image.new("RGB", img.size, (255, 255, 255))
    masked_img.paste(img, mask=mask)
    masked_img.save(os.path.join(data_root, "masked_images", f"{name}.png"))
    # break