#imports
import os, json, glob
import re
import matplotlib.pyplot as plt
from PIL import Image
import cv2 
import datetime
import logging  
import time
import numpy

############################################################ LOGS ############################################################ 

#Logger output should be Timestamp Class FunctionName AnythingExtraRelevant: Log message
FORMAT = '%(asctime)s %(class)s %(function)s %(info)s %(message)s'  

# dafault values for log 
default = {'class': 'class_undefined', 'function': 'function_undefined', 'info': 'No extra information'}
logging.basicConfig(
            level=logging.INFO,
            format=FORMAT,  
            handlers=[
                logging.FileHandler("processing.log"), #Save to a file called processing.log 
                logging.StreamHandler() #Show on screen.
            ]) 
logger=logging.getLogger()  


############################################################################################################################## 


############################################################ METADATA ############################################################ 
def process_source1(metadata, data):
    # for each label in labels.json
    for entry in data['data']:
        # if key is already on the dict then append label to labels
        if entry['filename'] in metadata:
            metadata[entry['filename']]['labels'].append(
                {
                "xmin": entry['xmin'],
                "ymin": entry['ymin'],
                "xmax": entry['xmax'],
                "ymax": entry['ymax'],
                "label": "Raccoon",
                "width_to_height": 0 
                })
        # else creat a new pair key value in dict for image and labels    
        else:
            metadata[entry['filename']] = {
                "path": entry['folder'] + entry['filename'],
                "filename": entry['filename'],
                "image_height": entry['image_height'],
                "image_width": entry['image_width'],
                "label_area_perc": 0,
                "labels": [
                {
                "xmin": entry['xmin'],
                "ymin": entry['ymin'],
                "xmax": entry['xmax'],
                "ymax": entry['ymax'],
                "label": "Raccoon",
                "width_to_height": 0 
                }]}

def process_source2(metadata, data):
    # for each label in labels.json
    for entry in data['data']:
        filename= entry['path'].split("/",1)[1] 
        # normalize coordinates
        xmin = entry['xmin']/int(entry['image_width'])
        xmax = entry['xmax']/int(entry['image_width'])
        ymin = entry['ymin']/int(entry['image_height'])
        ymax = entry['ymax']/int(entry['image_height'])
        # if key is already on the dict then append label to labels
        if filename in metadata:
            metadata[entry['filename']]['labels'].append(
                {
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "label": "Raccoon",
                "width_to_height": 0 
                })
        # else creat a new pair key value in dict for image and labels    
        else:
            metadata[filename] = {
                "path": entry['path'],
                "filename": filename,
                "image_height": int(entry['image_height']),
                "image_width": int(entry['image_width']),
                "label_area_perc": 0,
                "labels": [
                {
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "label": "Raccoon",
                "width_to_height": 0
                }]}

def normalize_cords(hor_cent,vert_cent,lab_w, lab_h, w, h):
    xmin = (hor_cent - lab_w/2)/w
    ymin = (vert_cent - lab_h/2)/h
    xmax = (hor_cent + lab_w/2)/w
    ymax = (vert_cent + lab_h/2)/h
    return xmin,ymin,xmax,ymax

def process_source3(metadata, data):
    # for each label in labels.json
    for entry in data['data']:
        path = entry['folder'] + entry['filename'] + '.jpg'
        filename = entry['filename'] + '.jpg'
        # fetch image shape from image
        im = Image.open('./sources/' + path)
        w, h = im.size
        # call auxiliary function to compute normalize coordinates given label center and dimensions
        xmin,ymin,xmax,ymax = normalize_cords(entry['label_horz_center'],entry['label_vert_center'],entry['label_width'],entry['label_height'],w,h)
        # if key is already on the dict then append label to labels
        if filename in metadata:
            metadata[filename]['labels'].append(
                {
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "label": "Raccoon",
                "width_to_height": 0 
                })
        # else creat a new pair key value in dict for image and labels      
        else:
            metadata[filename] = {
                "path": path,
                "filename": filename,
                "image_height": h,
                "image_width": w,
                "label_area_perc": 0,
                "labels": [
                {
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "label": "Raccoon",
                "width_to_height": 0 
                }]}

def process_sources(metadata, data, source_type):
    logger.info('Log: %s', 'Processing Source ' + str(source_type), extra=default|{'function': 'process_sources'})
    # process source according to type. In case source X is added: elif (...) process_sourceX(metadata, data)
    if source_type == 1:
        process_source1(metadata, data)
    elif source_type == 2:
        process_source2(metadata, data)
    else:
        process_source3(metadata, data)

def process_metadata():
    logger.info('Log: %s', 'Processing Metadata', extra=default|{'function': 'process_metadata'})
    # dict with all labels.json normalized to source format 1 with filename as key and processed image and label data as values
    metadata = {}
    sources = glob.glob("./sources/input_images_source*") 
    # process all existing sources
    for source in sources:
        json_path = source+'/labels.json'
        source_type = int(re.search(r'\d+', source).group())
        with open(json_path) as json_file:
            data = json.load(json_file)
            process_sources(metadata, data,source_type)
    return metadata

def save_metadata(metadata):
    logger.info('Log: %s', 'Saving Metadata', extra=default|{'function': 'save_metadata'})
    metadata_out = {}
    metadata_out['data'] = list(metadata.values())
    with open('data_output.json', 'w') as fp:
        json.dump(metadata_out, fp, indent=4)

def check_for_intersections(labels,w,h):
    intersection_perc = 0  
    if len(labels)> 1:
        for l in range(len(labels) -1): 
            for l2 in range(l+1,len(labels)):
                dx = min(labels[l]['xmax'], labels[l2]['xmax']) - max(labels[l]['xmin'], labels[l2]['xmin'])
                dy = min(labels[l]['ymax'], labels[l2]['ymax']) - max(labels[l]['ymin'], labels[l2]['ymin'])
                if (dx>0.0) and (dy>0.0):
                    intersection_perc += (dx*w*dy*h)
        return intersection_perc
    return intersection_perc

def label_perc(metadata):
    logger.info('Log: %s', 'Calculating percentage of the total area of the image occupied by labels', extra=default|{'function': 'label_perc'})
    for img in metadata:
        labels= metadata[img]['labels']
        l_size=0
        # compute total area occupied by labels
        for l in labels:
            l_size +=((l['xmax']-l['xmin'])*metadata[img]['image_width'])*((l['ymax']-l['ymin'])*metadata[img]['image_height']) 
        # compute area of intersection between labels
        intersections = check_for_intersections(metadata[img]['labels'],metadata[img]['image_width'],metadata[img]['image_height'])
        # update information in dict (total labels area - intersections)/ image area
        metadata[img]['label_area_perc'] = (l_size - intersections)/(metadata[img]['image_height']*metadata[img]['image_width'])

def plot_distribution(ratio_dist):
    logger.info('Log: %s', 'Plot the distribution of the output', extra=default|{'function': 'plot_distribution'})
    plt.figure(figsize=(5,4)) 
    plt.style.use('seaborn-whitegrid') 
    plt.hist(ratio_dist, bins=90, facecolor = '#2ab0ff', edgecolor='#169acf', linewidth=0.5)
    plt.title('Width to Height Distribution') 
    plt.xlabel('Bins') 
    plt.ylabel('Values') 
    plt.savefig('dist.png')

def width_to_height(metadata):
    logger.info('Log: %s', 'Calculating label width-to-height ratio of all the labels', extra=default|{'function': 'width_to_height'})
    ratio_dist = []
    # compute all labels ratio width to height
    for img in metadata:
        labels= metadata[img]['labels']
        for l in labels:
            l['width_to_height'] = (l['xmax']-l['xmin'])/(l['ymax']-l['ymin'])
            ratio_dist.append((l['xmax']-l['xmin'])/(l['ymax']-l['ymin']))
    # plot the distribution of the ratios
    plot_distribution(ratio_dist)

    
################################################################################################################################ 

############################################################ IMAGES ############################################################ 


def merge_orgimg_info(config_file_path):
    # open config file
    with open(config_file_path) as config_file:
        config = json.load(config_file)

    output_file = config['output_file']

    #fetch metadata file
    with open('./'+output_file) as metadata_file:
        info = json.load(metadata_file)
    
    #merge metadata file information and corresponding image in a dict 
    img_info = {}
    for entry in info['data']:
        im = Image.open('./sources/'+entry['path'])
        img_info['./sources/'+entry['path']] = {'img': im, 'info': entry}

    return img_info
    
def draw_labels(img_info):
    logger.info('Log: %s', 'Drawing Lables', extra=default|{'function': 'draw_labels'})
    # for each image draw the corresponding labels (info in dict)
    for path in img_info:
        image = cv2.imread(path)
        info = img_info[path]['info']
        for label in info['labels']:
            x_start = int(info['image_width'] * label['xmin'])
            y_start = int(info['image_height'] * label['ymax'])
            x_end = int(info['image_width'] * label['xmax'])
            y_end = int(info['image_height'] * label['ymin'])
            image = cv2.rectangle(image, (x_start, y_start),(x_end, y_end), (0, 0, 255), 2)
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # save in dict the image with drawn labels 
        img_info[path]['img_draw'] = Image.fromarray(img)

def crop_labels(img_info):
    logger.info('Log: %s', 'Cropping Lables', extra=default|{'function': 'crop_labels'})
    # for each image crop the corresponding labels (info in dict)
    for path in img_info:
        im = Image.open(path)
        info = img_info[path]['info']
        for label in info['labels']:
            x_start = int(info['image_width'] * label['xmin'])
            y_start = int(info['image_height'] * label['ymax'])
            x_end = int(info['image_width'] * label['xmax'])
            y_end = int(info['image_height'] * label['ymin'])
            
            im1 = im.crop((x_start, y_end, x_end, y_start))
            # save in dict the resulting images from cropping each label from the original image
            if ('img_croped' in img_info[path]):
                img_info[path]['img_croped'].append(im1) 
            else:
                img_info[path]['img_croped'] = [im1]
           
def add_padding(img_info,config_file_path):
    logger.info('Log: %s', 'Adding padding', extra=default|{'function': 'add_padding'})
    # Padding to ensure image size is consistently max size defined on config file.
    # open config file
    with open(config_file_path) as config_file:
        config = json.load(config_file)

    for path in img_info:
        for img in img_info[path]['img_croped']:   
            new_width = config['max_width']
            new_height = config['max_height']
            # resize image to size defined on config file keeping ratio
            img_aux = img.copy()
            img_aux.thumbnail((new_width, new_height), Image.ANTIALIAS)
            width, height = img_aux.size   
            # center image in top left corner      
            left = int( (new_width/2)- width/2)
            top = int( (new_height/2) - height/2)
            # black square
            result = Image.new(img_aux.mode, (new_width, new_height), (0, 0, 0))
    
            # paste image in the top left corner of the square (centered)
            result.paste(img_aux, (left, top))

            # add image with padding to dict
            if ('final_img' in img_info[path]):
                img_info[path]['final_img'].append(result) 
            else:
                img_info[path]['final_img'] = [result]

def save_imgs(img_info, config_file_path):
    logger.info('Log: %s', 'Saving images', extra=default|{'function': 'save_images'})
    #save final images in folder defined on the config file
    with open(config_file_path) as config_file:
        config = json.load(config_file)

    output_folder = config['output_folder']
    #create folder if it not exists
    os.makedirs('./'+output_folder, exist_ok=True)
    
    # clear folder
    for file in os.listdir('./'+output_folder):
        os.remove('./'+output_folder +'/'+file) 

    # save processed images as jpeg
    for img in img_info:
        count = 1
        filename=img_info[img]['info']['filename'].split('.')[0]
        for img_croped in img_info[img]['final_img']:
            path_img= './'+str(output_folder)+ '/'+filename+ '_'+str(count) +'.jpeg'
            img_croped.save(path_img, format="jpeg")
            count +=1
        
def aux_sliding_window(image, overlap, windowSize):
	for y in range(0, image.shape[0], overlap):
		for x in range(0, image.shape[1], overlap ):
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def sliding_window(img_info, config_file_path):
    logger.info('Log: %s', 'Showing sliding windows', extra=default|{'function': 'sliding_window'})
    with open(config_file_path) as config_file:
        config = json.load(config_file)

    # window size half of the defined in config file
    (winW, winH) = (int(config['max_width']/2), int(config['max_height']/2))
    # overlap as a funtion of the window size
    overlap  = int(config['overlap'] * (config['max_width']/2))
    #only sliding over first 2 images with labels drawn
    for img in list(img_info)[:2]:
        img_draw = img_info[img]['img_draw']
        img_draw = cv2.cvtColor(numpy.array(img_draw), cv2.COLOR_RGB2BGR)
        
        for (x, y, window) in aux_sliding_window(img_draw , overlap, windowSize=(winW, winH)):
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            clone = img_draw.copy()
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            cv2.imshow("Sliding Window", clone)
            cv2.waitKey(1)
            time.sleep(0.025)


############################################################################################################################## 


############################################################ MAIN ############################################################
def main():
    initial_time = datetime.datetime.now()  

    config_file_path = "./config.json"

    # 1. Process metadata file
    metadata = process_metadata()
    # Operations on the metadata following parsing:
    # Calculate percentage of the total area of the image occupied by labels.
    label_perc(metadata)
    # Calculate label width-to-height ratio of all the labels and plot the distribution of the output.
    width_to_height(metadata)
    #Save the output to a file
    save_metadata(metadata)

    # 2. Process images
    # Merge image and metadata info 
    img_info = merge_orgimg_info(config_file_path)
    # Draw labels on image
    draw_labels(img_info)
    # Crop image around label with size defined on config file (already stored in img_info)
    crop_labels(img_info)
    # Padding to ensure image size is consistently max size
    add_padding(img_info,config_file_path)
    # Save processed images as jpeg
    save_imgs(img_info, config_file_path)
    # Sliding window with labels drawn on the images
    sliding_window(img_info, config_file_path)

    final_time = datetime.datetime.now()
    print("Total time to run the script: ", (final_time -initial_time)) # 0:00:00.495917 without sliding window

if __name__ == "__main__":
    main()