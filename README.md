# BladeInsight_challenge
 
Run main script: 

python3 raccoon_processer.py

Run unit testing: 

python3 -m unittest test_raccoon_processer.py

Metadata dictionary format:

{'raccoon-3.jpg':
{
"path": "input_images_source_1/raccoon-3.jpg",
"filename": "raccoon-3.jpg",
"image_height": 480,
"image_width": 720,
"label_area_perc": 0.9882089120370371,
"labels": [
    {
        "xmin": 0.001388888888888889,
        "ymin": 0.0020833333333333333,
        "xmax": 1.0,
        "ymax": 0.9916666666666667,
        "label": "Raccoon",
        "width_to_height": 1.0091228070175438
    }
]
}
 
Image-information(img_info) dictionary format:

{'./sources/input_images_source_1/raccoon-3.jpg': 
{
'img': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=720x480 at 0x11B20A3E0>, 
'info': 
{
'path': 'input_images_source_1/raccoon-3.jpg',
'filename': 'raccoon-3.jpg', 
'image_height': 480, 
'image_width': 720, 
'label_area_perc': 0.9882089120370371, 
'labels': [
     {
         'xmin': 0.001388888888888889, 
         'ymin': 0.0020833333333333333, 
         'xmax': 1.0, 
         'ymax': 0.9916666666666667, 
         'label': 'Raccoon', 
         'width_to_height': 1.0091228070175438
      }
]
}, 
'img_draw': <PIL.Image.Image image mode=RGB size=720x480 at 0x1198E4280>, 
'img_croped': [<PIL.Image.Image image mode=RGB size=719x475 at 0x11B1D6860>], #can be more than one according to labels
'final_img': [<PIL.Image.Image image mode=RGB size=300x300 at 0x11B209C30>]
}
