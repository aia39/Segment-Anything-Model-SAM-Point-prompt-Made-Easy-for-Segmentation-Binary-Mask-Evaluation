# Author: Md. Abrar Istiak

# importing the module
import cv2
import glob
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import sys
sys.path.append("..")


from segment_anything import sam_model_registry, SamPredictor
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from google.colab.patches import cv2_imshow  #for colab


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--colab", type=str, default=False, help='colab or not')
ap.add_argument("-e", "--evaluate", type=int, default=False, help='evaluate or not')
ap.add_argument("-i", "--input", type=str, default='img/', help='Input directory')
ap.add_argument("-g", "--groundtruth", type=str, default='gt/', help='Ground truth directory')
ap.add_argument("-a", "--auto_seg", type=str, default=True, help='Automatic segmentation')
args = vars(ap.parse_args())

# function to display the coordinates of of the points clicked on the image (to choose foreground and background by click)
def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    
    ### To input foreground ###
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.circle(img, (x,y), 2, (255, 50, 0), 2)
        cv2.imshow('image', img)
        points.append([x,y])
        label.append(1)
        
    
    #### To input background ####
    if event == cv2.EVENT_RBUTTONDOWN:
        # displaying the coordinates on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.circle(img, (x,y), 2, (0, 50, 255), 2)
        cv2.imshow('image', img)
        points.append([x,y])
        label.append(0)



def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    return mask_image
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    



#### For metrics calculation ####
def calculateIoU(gtMask, predMask):
        # Calculate the true positives,
        # false positives, and false negatives
        tp = 0
        fp = 0
        fn = 0
 
        for i in range(len(gtMask)):
            for j in range(len(gtMask[0])):
                if gtMask[i][j] == True and predMask[i][j] == True:
                    tp += 1
                elif gtMask[i][j] == False and predMask[i][j] == True:
                    fp += 1
                elif gtMask[i][j] == True and predMask[i][j] == False:
                    fn += 1
 
        # Calculate IoU
        iou = tp / (tp + fp + fn)
        dice = 2*tp/(2* tp + fp + fn)
        return iou, dice


       

# evaluation_mode = True   #keep it false if you don't want to evaluate against ground truth
# colab = True    ##keep it false if you are using local pc environment

if args["colab"]:
	!pip install git+https://github.com/facebookresearch/segment-anything.git   ##run it to install git repo

	!wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O sam_vit_h.pth  ##download the SAM trained weights

	!wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O sam_vit_b.pth  ##download the SAM trained weights


cur = os.getcwd()
datapath = os.path.join(cur,'*.jpg')
all_files = glob.glob(datapath)
all_points_coord = []
all_points_label = []

for i in range(len(all_files)):
    # reading the image
    img = cv2.imread(all_files[i], 1)

    points = []
    label = []
    
    cv2.imshow('image', img)

    # setting mouse handler for the image and calling the click_event() function
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    all_points_coord.append(points)


cur = os.getcwd()
folder = args["input"]
types = ('*.png', '*.jpg')   #put your file extension
all_files = []
for files in types:
    all_files.extend(glob.glob(os.path.join(folder, files)))


if args["evaluate"]:
	### Loading Ground truth mask ###
	folder = args["groundtruth"]
	types = ('*.png', '*.jpg')
	all_files_gt = []
	for files in types:
	    #all_files.extend(glob.glob(os.path.join(cur, files)))
	    all_files_gt.extend(glob.glob(os.path.join(folder, files)))

	gt_all = []

	for i in range(len(all_files_gt)):
	    gt_image = cv2.imread(all_files_gt[i], cv2.IMREAD_GRAYSCALE)
	    #gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
	    
	    #gt_image = cv2.resize(gt_image, (512, 512))   #for bigger image
	    
	    
	    gt_image = gt_image > 200  #200    #binarizing
	    gt_all.append(gt_image)
	    #gt_all = np.append(gt_all, gt_image) 


###################################################################
##### Segmentation on an image directory with point prompts #######
###################################################################

sam_checkpoint = "sam_vit_h.pth"   ##choose the type of weights that used
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

if not os.path.exists('outputs_prompt'):
    os.makedirs('outputs_prompt')

total_iou = 0
total_dice = 0

for i in range(len(all_points[0])):
    image = cv2.imread(all_files[i])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.axis('on')
    plt.show()
    
    predictor = SamPredictor(sam)

    #image = cv2.resize(image, (512, 512))  #for bigger image

    predictor.set_image(image)
        
    input_point = np.array(all_points[0][i])
    
    input_label = all_points[1][i]

  
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask

    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        mask_input=mask_input[None, :, :],
        multimask_output=False,
    )
    
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    overlaid_mask = show_mask(masks, plt.gca())

    
    final_mask = np.squeeze(masks)*255
    cv2_imshow(final_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    image_name = all_files[i].split(os.path.sep)[-1]
    cv2.imwrite(os.path.join('outputs_prompt', image_name), final_mask)
    
    
    if args["evaluate"]:
	    # Evaluating metrics
	    single_iou, single_dice = calculateIoU(gt_all[i], np.squeeze(masks))
	    total_iou += single_iou
	    total_dice += single_dice  


if args["evaluate"]:
	iou = total_iou/len(all_files)
	print(f"Total IOU in the dataset is {iou}")

	dice = total_dice/len(all_files)
	print(f"Total Dice coeff in the dataset is {dice}")


if args["auto_seg"]:
	#########################################################
	##### Automatic Segmentation on an image directory ######
	#########################################################
	if not os.path.exists('outputs_auto'):
	os.makedirs('outputs_auto')

	def show_anns(anns):
	    if len(anns) == 0:
	        return
	    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
	    #ax = plt.gca()
	    #ax.set_autoscale_on(False)
	    bool_mask = np.zeros_like(sorted_anns[0], dtype = 'bool')
	    for ann in sorted_anns:
	        m = ann['segmentation']
	        img = np.ones((m.shape[0], m.shape[1], 3))
	        for i in range(3):
	            img[:,:,i] = 0
	        #np.dstack((img, m*1))
	        #ax.imshow(np.dstack((img, m*1)))
	        bool_mask = bool_mask|m
	    return bool_mask


	mask_generator = SamAutomaticMaskGenerator(sam)

	total_iou = 0
	total_dice = 0

	for i in range(len(all_files)):
	  image = cv2.imread(all_files[i])
	  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	  masks = mask_generator.generate(image)
	  
	  final_mask1 = show_anns(masks)
	  
	  final_mask2 = final_mask1*255
	  cv2_imshow(final_mask2)
	  cv2.waitKey(0)
	  cv2.destroyAllWindows()

	  image_name = all_files[i].split(os.path.sep)[-1]
	  cv2.imwrite(os.path.join('outputs_auto', image_name), final_mask2)
	  
	  if args["evaluate"]:
		  # Evaluating metrics
		  single_iou, single_dice = calculateIoU(gt_all[i], np.squeeze(final_mask1))
		  total_iou += single_iou
		  total_dice += single_dice 

	if args["evaluate"]:
		iou = total_iou/len(all_files)
		print(f"Total IOU in the dataset is {iou}")

		dice = total_dice/len(all_files)
		print(f"Total Dice coeff in the dataset is {dice}")

