# Semantic-segmentation-of-adherent-cells-
This is designed for cell counting in adherent cell images and works with images that are >= 256x256 pixels.
Input: bright-field cell images
Output: cell counts
*It can be used for any image size, as long as the size is no smaller than 256x256 pixels.

Experimental:
1. culutre MDA-MB-231 breast cancer cells in a well-plate
2. cells were stained with Calcein AM before imaging.
3. cells were scaned under a 20X magnification microscope in brightfield and EGFB channel.
4. cell image: bright-field image; cell mask: thresholded fluorescent image
![image](https://github.com/user-attachments/assets/1ef64960-b59a-4e82-bbda-4ee2f34a6d53)

Training and cell counting:
1. semantic segmentation of adherent cells
  -Training: Uses standard U-Net framework 
  -Training: Patches of 256x256 from cell images and masks 
2. Smooth blending of patches for semantic segmentation of large images
3. Identify each cells using StarDist model (cell counting)
![image](https://github.com/user-attachments/assets/61dcd355-714a-4fed-835b-fca55d7a56f5)


