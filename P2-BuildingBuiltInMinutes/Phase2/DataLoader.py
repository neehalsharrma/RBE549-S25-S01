import os
import json
import numpy as np
import cv2


class DataLoader:
    def __init__(self, data_path= "nerf_synthetic/lego"):
        self.data_path = data_path

    def loadDataset(self, mode):
        """
        Input:
            data_path: dataset path
            mode: train or test
        Outputs:
            camera_info: image width, height, focus
            images: images
            pose: corresponding camera pose in world frame
        """
        image_base_path= self.data_path + "/" 
        jsonfile_path= self.data_path + "/transforms_"+ mode +".json"

        with open(jsonfile_path) as jsonfile:
            data = json.load(jsonfile)

        camera_angle_x= data["camera_angle_x"]
        images= []
        poses= []

        for i in range(len(data["frames"])): 
            frame= data["frames"][i]
            image_path= os.path.join(image_base_path, frame["file_path"]+".png")
            img= cv2.imread(image_path)
            images.append(img)
            pose= frame["transform_matrix"]
            poses.append(pose)

        images= np.array(images)
        poses= np.array(poses)

        # shape returns height, width, channel (so select first two)
        H,W= images[0].shape[:2]
        focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

        # Assuming the same camera is being used for every image, the H,W and focal length will be the same for all images 
        camera_info= [W, H,focal]
        return images, poses, camera_info
        

# For Testing 
# if __name__ == "__main__":
#     # load data
#     print("Loading data...")
#     DataLoader= DataLoader()
#     images, poses, camera_info = DataLoader.loadDataset("train")
#     print("Printing camera info")
#     print(camera_info)
#     print("Poses", poses)
#     print(len(images), len(poses))
