from PIL import Image
import skimage.io as io
import os

#this function reads in images and then resizes them for our eventual cnn
def resize_images(path,dest):
    extracted_path = path[0:-6]
    names=os.listdir(extracted_path)
    images=io.ImageCollection(path)

    for i in range(len(images)):
        img1=Image.open(str(extracted_path)+'/'+str(names[i]))
        new_name = 'resized_'+str(names[i])
        res1=img1.resize((500,500),resample=0)
        res1.save(str(dest)+'/'+str(new_name))
    
    return 'Images resized!'

path = r'0_not_ditylum/*.tif'
dest = r'0_resized'
resize_images(path,dest)
