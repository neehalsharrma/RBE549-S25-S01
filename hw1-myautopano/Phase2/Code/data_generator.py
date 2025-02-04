import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import time
import re
import concurrent.futures

# run code from the repo directory
relative_path="hw1-myautopano\Phase2\Data\Train\\"
label_file_name="..\TrainLabels.txt"
time_read_write=0
total_time=0

def getListImages(image_path):
    image_directory = os.path.dirname(image_path)

    # List all files in the directory
    all_files = os.listdir(image_directory)
    # Filter out the image files (assuming they are .jpg files)
    image_files = [f for f in all_files if f.endswith(".jpg")]
    sorted_files = sorted(image_files, key=lambda x: int(re.search(r'\d+', x).group()))
    return sorted_files

def getBatch(image_list, batchNum, size):
    tot= len(image_list)
    return image_list[batchNum*size:min((batchNum+1)*size, tot)]

def getImages(image_list):
    global time_read_write
    images=[]
    start_time=time.time()
    for image_file in image_list:
        images.append(cv2.imread(relative_path+image_file))
    end_time= time.time()
    time_read_write+=end_time-start_time
    return images

def getPertubVals(p):
    return [random.randint(-p,p) for i in range(8)]

def getCoordVals(p, h, w, patch_size):
    # l= [random.randint(200,h-(2*p)) for i in range(2)]
    # l.extend([random.randint(200,w-(2*p)) for i in range(2)])
    limitx= w-patch_size-(2*p)
    limity= h-(2*p)-patch_size
    if(limitx<2*p or limity<2*p):
        return None
    x1= random.randint(2*p, limitx)
    y1= random.randint(2*p, limity)
    l=[x1, x1+patch_size, y1, y1+patch_size]
    return l

def getRandomPatches(img, p, patch_size, iname, patches_per_image):
    # cv2.imshow("IMG", img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    h, w, _ = img.shape
    aHab=[] #List of tildaH values
    for i in range(patches_per_image):
        # Image coordinates
        # (h1, w1) ... (h2, w2)
        # (h3, w3) ... (h4, w4)
        coords= getCoordVals(p, h, w, patch_size)
        while coords is None:
            coords= getCoordVals(p*2//3, h, w, patch_size) 
        w1, w4, h1, h4=coords
        ph1, ph2, ph3, ph4, pw1,  pw2, pw3, pw4 = getPertubVals(p)
        # P1, P2, P3, P4
        ca = np.float32([[w1, h1], [w1, h4], [w4, h1], [w4, h4]])
        cb = np.float32([[w1+pw1, h1+ph1], [w1+pw2, h4+ph2], [w4+pw3, h1+ph3], [w4+pw4, h4+ph4]])

        Hab= cv2.getPerspectiveTransform(ca, cb)
        Hba= np.linalg.inv(Hab)
        Pa= img[h1:h4, w1:w4]
        transformed_img= cv2.warpPerspective(img, Hba, (w,h) )
        Pb= transformed_img[h1:h4, w1:w4]
        aHab= np.subtract(cb,ca)
        # cv2.imshow("Pa", Pa)
        # cv2.imshow("Pb", Pb)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        saveRandomPatches(Pa, Pb, iname+"_"+str(i+1))
    return aHab


def saveRandomPatches(Pa, Pb, iname):
    global time_read_write
    # to start saving from 1
    start_time= time.time()
    cv2.imwrite(str(relative_path+"processed\\"+iname + "A.jpg"), Pa)
    cv2.imwrite(str(relative_path+"processed\\"+iname + "B.jpg"), Pb)
    end_time= time.time()
    time_read_write+=end_time-start_time

def generate_images_batch(p, batch_images, batch_num, batch_size, patch_size, patches_per_image):
    i=0
    transformation_list=[]
    for image in batch_images:
        i+=1
        aHab= getRandomPatches(image, p, patch_size, str(batch_num*batch_size+i), patches_per_image)
        transformation_list.extend(aHab)
    return transformation_list

def save_list_to_file(data, filename):
    """Saves a list of lists to a text file with line breaks and comma-separated values."""
    with open(filename, "w") as file:
        for line in data:
            file.write(",".join(map(str, line)) + "\n")

# Single threaded function
# def generate_images(p, relative_path, batch_size, patch_size, patches_per_image):
#     image_list= getListImages(relative_path)
#     transformation_list=[]
#     iterations= (len(image_list)+batch_size-1)//batch_size #Adding batch_size-1 so the division ceils
#     for i in range(iterations):
#         print("On batch- ", i)
#         batch_list= getBatch(image_list, i, batch_size)
#         batch_images= getImages(batch_list)
#         transformation_list.extend(generate_images_batch(p, batch_images, i, batch_size, patch_size, patches_per_image))

#     save_list_to_file(transformation_list, relative_path+label_file_name)


def generate_images(p, relative_path, batch_size, patch_size, patches_per_image):
    image_list = getListImages(relative_path)
    transformation_list = []
    
    iterations = (len(image_list) + batch_size - 1) // batch_size  # Ceiling division
    
    def process_batch(i):
        """Function to process a single batch"""
        print("On batch- ", i)
        batch_list = getBatch(image_list, i, batch_size)
        batch_images = getImages(batch_list)
        return generate_images_batch(p, batch_images, i, batch_size, patch_size, patches_per_image)

    # Use ThreadPoolExecutor for multithreading
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_batch, range(iterations)))

    # Flatten the list of results
    for result in results:
        transformation_list.extend(result)

    save_list_to_file(transformation_list, relative_path + label_file_name)


start= time.time()
# p, image path, batch_size, patch_size, patches_per_image
generate_images(30, relative_path, 50, 128, 2)
end= time.time()
total_time= end- start

print("Total time- ", total_time, "    time_read_write- ", time_read_write)