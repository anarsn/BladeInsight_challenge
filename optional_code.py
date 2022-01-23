import logging  

############################################################ LOGS ############################################################ 

class Log:
    def __init__(self, function_name = 'function_undefined', message ='message_undefined', class_name ='class_undefined', extra_info='No extra information', timestamp = datetime.datetime.now()):
        self.timestamp = timestamp
        self.class_name = class_name
        self.function_name = function_name 
        self.message = message
        self.extra_info = extra_info
    
    def print_log(self):
        # Show log on screen
        log = str(self.timestamp) +' '+ self.class_name +' '+ self.function_name+' '+  self.extra_info + ': Log ' + self.message + '\n'
        print(log)
        # Open file in append mode
        file_object = open('processing.log', 'a')
        # Writes to file
        file_object.write(log)
        # Close the file
        file_object.close()

############################################################################################################################## 

def label_perc_temp(height,width,labels):
    l_size=0
    for l in labels:
        l_size +=((l['xmax']-l['xmin'])*width)*((l['ymax']-l['ymin'])*height)
    return l_size/(height*width)

def width_to_height_temp(xmin,xmax,ymin,ymax):
    return (xmax-xmin)/(ymax-ymin)
