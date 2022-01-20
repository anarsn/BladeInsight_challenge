#imports
from importlib.metadata import metadata
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
import os, json, glob
import re
import matplotlib.pyplot as plt
from PIL import Image
import cv2 



def label_perc_temp(height,width,labels):
    l_size=0
    for l in labels:
        l_size +=((l['xmax']-l['xmin'])*width)*((l['ymax']-l['ymin'])*height)
    return l_size/(height*width)

def width_to_height_temp(xmin,xmax,ymin,ymax):
    return (xmax-xmin)/(ymax-ymin)


############################################################ METADATA ############################################################ 
def process_source1(metadata, data):
    for entry in data['data']:
        if entry['filename'] in metadata:
            metadata[entry['filename']]['labels'].append(
                {
                "xmin": entry['xmin'],
                "ymin": entry['ymin'],
                "xmax": entry['xmax'],
                "ymax": entry['ymax'],
                "label": "Raccoon",
                "width_to_height": 0 #width_to_height(entry['xmin'],entry['xmax'],entry['ymin'],entry['ymax'])
                })
            
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
                "width_to_height": 0 #width_to_height(entry['xmin'],entry['xmax'],entry['ymin'],entry['ymax'])
                }]}
    return metadata


def process_source2(metadata, data):
    for entry in data['data']:
        filename= entry['path'].split("/",1)[1] 
        xmin = entry['xmin']/int(entry['image_width'])
        xmax = entry['xmax']/int(entry['image_width'])
        ymin = entry['ymin']/int(entry['image_height'])
        ymax = entry['ymax']/int(entry['image_height'])
        if filename in metadata:
            metadata[entry['filename']]['labels'].append(
                {
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "label": "Raccoon",
                "width_to_height": 0 #width_to_height(xmin,xmax,ymin,ymax)
                })
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
                "width_to_height": 0 #width_to_height(xmin,xmax,ymin,ymax)
                }]}
    return metadata

def normalize_cords(hor_cent,vert_cent,lab_w, lab_h, w, h):
    xmin = (hor_cent - lab_w/2)/w
    ymin = (vert_cent - lab_h/2)/h
    xmax = (hor_cent + lab_w/2)/w
    ymax = (vert_cent + lab_h/2)/h
    return xmin,ymin,xmax,ymax

def process_source3(metadata, data):
    for entry in data['data']:
        path = entry['folder'] + entry['filename'] + '.jpg'
        filename = entry['filename'] + '.jpg'
        im = Image.open('./sources/' + path)
        w, h = im.size
        xmin,ymin,xmax,ymax = normalize_cords(entry['label_horz_center'],entry['label_vert_center'],entry['label_width'],entry['label_height'],w,h)
        if filename in metadata:
            metadata[filename]['labels'].append(
                {
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "label": "Raccoon",
                "width_to_height": 0 #width_to_height()
                })
            
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
                "width_to_height": 0 #width_to_height()
                }]}
        metadata[filename]['label_area_perc'] = 0
    return metadata

def process_sources(metadata, data, source_type):
    print("Processing source" , source_type)
    if source_type == 1:
        return process_source1(metadata, data)
    elif source_type == 2:
        return process_source2(metadata, data)
    else:
        return process_source3(metadata, data)


def process_metadata():
    metadata = {}
    sources = glob.glob("./sources/input_images_source*") 
    for source in sources:
        json_path = source+'/labels.json'
        source_type = int(re.search(r'\d+', source).group())
        with open(json_path) as json_file:
            data = json.load(json_file)
            metadata = metadata | process_sources(metadata, data,source_type)
    return metadata

def save_metadata(metadata):
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
    for img in metadata:
        labels= metadata[img]['labels']
        l_size=0
        for l in labels:
            l_size +=((l['xmax']-l['xmin'])*metadata[img]['image_width'])*((l['ymax']-l['ymin'])*metadata[img]['image_height']) #round(_,1)?
        intersections = check_for_intersections(metadata[img]['labels'],metadata[img]['image_width'],metadata[img]['image_height'])
        metadata[img]['label_area_perc'] = (l_size - intersections)/(metadata[img]['image_height']*metadata[img]['image_width'])

def plot_distribution(ratio_dist):
    plt.figure(figsize=(5,4)) 
    plt.style.use('seaborn-whitegrid') 
    plt.hist(ratio_dist, bins=90, facecolor = '#2ab0ff', edgecolor='#169acf', linewidth=0.5)
    plt.title('Width to Height Distribution') 
    plt.xlabel('Bins') 
    plt.ylabel('Values') 
    plt.savefig('dist.png')

def width_to_height(metadata):
    ratio_dist = []
    for img in metadata:
        labels= metadata[img]['labels']
        for l in labels:
            l['width_to_height'] = (l['xmax']-l['xmin'])/(l['ymax']-l['ymin'])
            ratio_dist.append((l['xmax']-l['xmin'])/(l['ymax']-l['ymin']))
    plot_distribution(ratio_dist)

    
################################################################################################################################ 

############################################################ IMAGES ############################################################ 


def merge_orgimg_info(config_file_path):
    with open(config_file_path) as config_file:
        config = json.load(config_file)

    output_file = config['output_file']
    with open('./'+output_file) as metadata_file:
        info = json.load(metadata_file)
    

    img_info = {}
    for entry in info['data']:
        im = Image.open('./sources/'+entry['path'])
        img_info['./sources/'+entry['path']] = {'img': im, 'info': entry}
    """
    ds = tf.keras.utils.image_dataset_from_directory("./sources")
    print(ds.class_names)
    plt.figure(figsize=(10, 10))
    for images, labels in ds.take(1):
        for i in range(4):
            ax = plt.subplot(2, 2, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(ds.class_names[labels[i]])
            plt.axis("off")
    plt.savefig('ola.png')
    """
    #raw = tf.io.read_file("./input_images_source_2/raccoon-125.png")
    #image = tf.image.decode_png(raw, channels=3)
    # the `print` executes during tracing.
    #print("Initial shape: ", image.shape)
    #builder = tfds.ImageFolder('./input_images_source_2')
    #print(builder.info)
    return img_info
    
def draw_labels(img_info):
    for path in img_info:
        image = cv2.imread(path)
        info = img_info[path]['info']
        for label in info['labels']:
            x_start = int(info['image_width'] * label['xmin'])
            y_start = int(info['image_height'] * label['ymax'])
            x_end = int(info['image_width'] * label['xmax'])
            y_end = int(info['image_height'] * label['ymin'])
            image = cv2.rectangle(image, (x_start, y_start),(x_end, y_end), (0, 0, 255), 2)
            ### cv2-BGR , pill 0 RGB
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_info[path]['img_draw'] = Image.fromarray(img)
        #cv2.imwrite('ola.jpg', image)
    return img_info

def crop_labels(img_info):
    for path in img_info:
        im = Image.open(path)
        info = img_info[path]['info']
        for label in info['labels']:
            x_start = int(info['image_width'] * label['xmin'])
            y_start = int(info['image_height'] * label['ymax'])
            x_end = int(info['image_width'] * label['xmax'])
            y_end = int(info['image_height'] * label['ymin'])
            
            # Cropped image of above dimension
            # (It will not change original image)
            im1 = im.crop((x_start, y_end, x_end, y_start))
            if ('img_croped' in img_info[path]):
                img_info[path]['img_croped'].append(im1) 
            else:
                img_info[path]['img_croped'] = [im1]
           
    return img_info

def save_imgs(img_draw, config_file_path):
    with open(config_file_path) as config_file:
        config = json.load(config_file)

    output_folder = config['output_folder']
    os.makedirs('./'+output_folder, exist_ok=True)
    count=1
    for img in img_draw:
        path_img= './'+str(output_folder)+ '/'+ str(count) +'.jpeg'
        img_draw[img]['img_draw'].save(path_img, format="jpeg")
        count+=1
    count=16
    for img in img_draw:
        print(img_draw[img]['img_croped'])
        for img_croped in img_draw[img]['img_croped']:
            path_img= './'+str(output_folder)+ '/'+ str(count) +'.jpeg'
            print(img_croped)
            img_croped.save(path_img, format="jpeg")
            count+=1

################################################################################################################################ 


############################################################ MAIN ############################################################
def main():
    config_file_path = "./config.json"
    metadata = process_metadata()
    label_perc(metadata)
    width_to_height(metadata)
    save_metadata(metadata)
    img_info = merge_orgimg_info(config_file_path)
    img_info_draw = draw_labels(img_info)
    img_crop = crop_labels(img_info_draw)
    save_imgs(img_crop, config_file_path)
    

if __name__ == "__main__":
    main()