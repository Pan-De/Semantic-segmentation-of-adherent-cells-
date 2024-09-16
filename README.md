# Semantic-segmentation-of-adherent-cells-

Training and testing for semantic segmentation (Unet) of adherent cells (MDA-MB-231 breast cancer cells)

1. Uses standard Unet framework
2. Patches of 256x256 from images and masks
3. Trained model can be used on an image with a size larger than 256X256
4. Training images were obtained from 20X magnification microscope image scans in brightfield and EGFB channel.
5. MDA-MB-231 breast cancer cells were stained with Calcein AM for mask generation.
