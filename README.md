# Semantic-segmentation-of-adherent-cells-
This is designed for cell counting in adherent cell images. <br />
Input: bright-field cell images.<br />
Output: cell counts.<br />
*It can be used for any image size, as long as the size is no smaller than 256x256 pixels.

Experimental:  
1. Culutre MDA-MB-231 breast cancer cells in a microwell plate.
2. Cells were stained with Calcein AM before imaging.
3. Cells were scaned under a 20X magnification microscope in brightfield and EGFB channel.
4. Cell image: bright-field image; cell mask: thresholded fluorescent image
![image](https://github.com/user-attachments/assets/d9993254-a067-40ee-9aa0-f9190ec39912)


Training and cell counting:

1. Semantic segmentation of adherent cells.    
1.1Training: Uses standard U-Net framework.   
1.2Training: Patches of 256x256 from cell images and masks.  
   
2. Smooth blending of patches for semantic segmentation of large images.
  
3. Identify each cells using StarDist model (cell counting).
![image](https://github.com/user-attachments/assets/61dcd355-714a-4fed-835b-fca55d7a56f5)


