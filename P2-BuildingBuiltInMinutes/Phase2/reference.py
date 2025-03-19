import argparse
import glob
from tqdm import tqdm
import random
# from torch.utils.tensorboard import SummaryWriter
import imageio
import torch
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import json
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torchmetrics.image import StructuralSimilarityIndexMeasure

from NeRFModel import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
torch.random.manual_seed(0)

def load_images(image_dir):
    images = []
    for i, filename in enumerate(os.listdir(image_dir)):
        img_path = os.path.join(image_dir, f"r_{i}.png")
        img = cv2.imread(img_path) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (400, 400)) / 255
        images.append(img)
        if i == 199:
            break

    return images

def loadDataset(data_path, mode, device):
    with open(data_path, 'r') as file:
        data = json.load(file)

    camera_angle_x = data['camera_angle_x']
    frames = data['frames']
    
    file_paths = []
    rotations = []
    poses = []

    for frame in frames:
        file_paths.append(frame['file_path'])
        rotations.append(frame['rotation'])
        poses.append(frame['transform_matrix'])
    
    image_dir = os.path.join(os.path.dirname(data_path), 'train')
    images = load_images(image_dir)
    images = np.array(images, dtype=np.float32)

    # plot image
    plt.imshow(images[0])
    plt.show()

    images = torch.from_numpy(images).to(device)

    poses = np.array(poses, dtype=np.float32)
    poses = torch.from_numpy(poses).to(device)

    H, W = images[0].shape[0], images[0].shape[1]
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    camera_info = (H, W, focal)

    test_data_path = "Phase2/Data/ship/transforms_test.json"
    #os.path.join(os.path.dirname(data_path), 'test', 'transforms_test.json')
    with open(test_data_path, 'r') as file:
        test_data = json.load(file)

    test_frames = test_data['frames']
    test_file_paths = []
    test_rotations = []
    test_poses = []

    for frame in test_frames:
        test_file_paths.append(frame['file_path'])
        test_rotations.append(frame['rotation'])
        test_poses.append(frame['transform_matrix'])

    test_image_dir = os.path.join(os.path.dirname(data_path), 'test')
    test_images = load_images(test_image_dir)
    test_images = np.array(test_images, dtype=np.float32)
    test_images = torch.from_numpy(test_images)

    test_poses = np.array(test_poses, dtype=np.float32)
    test_poses = torch.from_numpy(test_poses)
    # check shapes
    print(images.shape)
    print(test_images.shape)
    print(poses.shape)
    print(test_poses.shape)
    # test_poses = None
    # test_images = None

    return camera_info, images, poses, test_poses, test_images

def PixelToRay(camera_info, pose):
    """
    Input:
        camera_info: image width, height, camera matrix
        pose: camera pose in world frame
        pixelPoition: pixel position in the image
        args: get near and far range, sample rate ...
    Outputs:
        ray origin and direction
    """
    H, W, focal = camera_info

    mesh_x, mesh_y = torch.meshgrid(torch.linspace(0, W-1, W).to(pose), torch.linspace(0, H-1, H).to(pose), indexing='ij')

    # transpose needed as mesh is for vector representation wrt to the center of the image. therefore all columns of x need to be 0    

    mesh_x = mesh_x.T
    mesh_y = mesh_y.T

    x = (mesh_x - W/2) / focal
    y = (mesh_y - H/2) / focal
    
    directions = torch.stack((x, -y, -torch.ones_like(x)), dim=-1)
    directions = directions[..., np.newaxis, :]

    rotation = pose[:3, :3]
    translation = pose[:3, -1].view(1, 1, 3)
    
    ray_direction = torch.sum(directions*rotation, dim=-1)
    ray_direction = ray_direction/torch.linalg.norm(ray_direction, axis=-1, keepdim=True)
    ray_origin = translation.expand(ray_direction.shape[0], ray_direction.shape[1], -1)

    return ray_direction, ray_origin

def generateBatch(ray_origins, ray_directions, gt_colors, args, train=True):
    """
    Input:
        images: all images in dataset
        poses: corresponding camera pose in world frame
        camera_info: image width, height, camera matrix
        args: get batch size related information
    Outputs:
        A set of rays origins, directions and gt colors
    """
    if train:
        indices = np.random.choice(ray_origins.shape[0], 4096, replace=False)
        sample_rays_o = ray_origins[indices].to(device)
        sample_rays_d = ray_directions[indices].to(device)
        sample_colors = gt_colors[indices].to(device)
    else:
        sample_rays_o = ray_origins.to(device)
        sample_rays_d = ray_directions.to(device)
        sample_colors = gt_colors.to(device)
    
    return sample_rays_o, sample_rays_d, sample_colors

def generateRays_and_gt(images, poses, camera_info, args):
    """
    Input:
        images: all images in dataset
        poses: corresponding camera pose in world frame
        camera_info: image width, height, camera matrix
        args: get batch size related information
    Outputs:
        A set of rays origins, directions and gt colors
    """
    all_rays_o = []
    all_rays_d = []
    all_colors = []

    H, W, focal = camera_info
    for i in range(images.shape[0]):
        img = images[i]
        pose = poses[i]

        rays_d, rays_o = PixelToRay(camera_info, pose)
        all_rays_o.append(rays_o.view(-1, 3))
        all_rays_d.append(rays_d.view(-1, 3))
        all_colors.append(img.view(-1, 3))

    all_rays_o = torch.cat(all_rays_o, dim=0)
    all_rays_d = torch.cat(all_rays_d, dim=0)
    all_colors = torch.cat(all_colors, dim=0)

    return all_rays_o, all_rays_d, all_colors


def render(model, rays_origin, rays_direction, args):
    """
    Input:
        model: NeRF model
        rays_origin: origins of input rays
        rays_direction: direction of input rays
    Outputs:
        rgb values of input rays
    """
    n_bins = 192
    t_near = 2
    t_far = 6
    
    t = torch.linspace(t_near, t_far, n_bins)
    t = t.expand(rays_origin.shape[0], n_bins).clone().to(device)
    
    random_offsets = torch.rand(*rays_origin.shape[:-1], n_bins) * (t_far - t_near) / n_bins
    t += random_offsets.to(device)

    query_input = rays_origin.unsqueeze(1) + rays_direction.unsqueeze(1) * t.unsqueeze(-1)
    flattened_query_input = query_input.view(-1, 3)
    
    rays_direction = rays_direction.expand(n_bins, rays_direction.shape[0], 3).transpose(0, 1).reshape(-1, 3)
    
    # colors, sigma = model(flattened_query_input)
    colors, sigma = model(flattened_query_input, rays_direction)

    colors = colors.view(*query_input.shape[:-1], 3)
    sigma = sigma.view(*query_input.shape[:-1])
    
    # Use Volume rendering to get the final image
    dists = torch.zeros_like(t)
    # Fill in the calculated distances into the new tensor, leaving the last entry as zero
    dists[..., :-1] =  t[..., 1:] - t[..., :-1]
    # Now, set the last entry to 1e10 to simulate an "infinite" distance for the last sample
    dists[..., -1] = 1e10
    

    # Compute alpha values from sigma and distances and Compute weights for RGB and depth accumulation
    alpha = 1.0 - torch.exp(-sigma * dists)
    adjusted_alpha = 1.0 - alpha + 1e-10

    # Pad the tensor with ones at the beginning of the dimension of interest
    padded_alpha = torch.cat([torch.ones(*adjusted_alpha.shape[:-1], 1).to(device), adjusted_alpha], dim=-1)

    # Perform the cumulative product on the padded tensor
    cumprod_padded = torch.cumprod(padded_alpha, dim=-1)

    # Remove the first element to get the "exclusive" cumulative product
    T = cumprod_padded[..., 1:]
    weights = alpha * T

    # Compute weighted sum of colors to get RGB map
    rgb_map = torch.sum(weights.unsqueeze(-1) * colors, dim=-2)
    
    # Compute weighted sum of z values to get depth map
    depth_map = torch.sum(weights * t, dim=-1)

    # Compute sum of weights to get accumulated map
    acc_map = torch.sum(weights, dim=-1)

    return rgb_map
    
        
def visualize_rays(ray_direction, ray_origin):

    distance = 0.1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Subsample the rays for plotting
    for i in range(0, ray_direction.shape[0], 10):
        for j in range(0, ray_direction.shape[1], 10):
            
            # Plot a line from the ray origin to the end point
            ax.quiver(ray_origin[i, j, 0].item(), ray_origin[i, j, 1].item(), ray_origin[i, j, 2].item(),
                    ray_direction[i, j, 0].item(), ray_direction[i, j, 1].item(), ray_direction[i, j, 2].item(),
                    length=distance, normalize=True)

    # Set labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('Ray Directions at Distance ' + str(distance))

    # Show the plot
    plt.show()

def loss(groundtruth, prediction):
    return F.mse_loss(groundtruth, prediction)
    

def train(images, poses, camera_info, args):

    model = NeRFmodel().to(device)
    # model = VeryTinyNerfModel().to(device)
    # model.load_state_dict(torch.load("Output/checkpoint/model_500.pt", map_location=device))
    # model = TinyNeRFmodel().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lrate)
    scaler = GradScaler()
    
    Loss = []
    Epochs = []
        
    # read all images, poses and generate rays and gt colors
    ray_origins, ray_directions, gt_colors = generateRays_and_gt(images, poses, camera_info, args)
    print(f" Ray Origins: {ray_origins.shape}")
    print(f" Ray Directions: {ray_directions.shape}")
    print(f" GT Colors: {gt_colors.shape}")

    best_loss = float('inf')
    for i in range(args.max_iters):

    # for i in range(args.epochs):
        model.train()

        with autocast():
        
            batch_ray_origins, batch_ray_directions, batch_gt_colors = generateBatch(ray_origins, ray_directions, gt_colors, args)
            # print(f"gt_colors shape: {batch_gt_colors.shape}")
            rgb_pred = render(model, batch_ray_origins, batch_ray_directions, args)
            # rgb_pred = render_rays(model, batch_ray_origins, batch_ray_directions)
            # print(f" RGB shape: {rgb_pred.shape}")
            
            current_loss = loss(batch_gt_colors, rgb_pred)
            print(f" Iteration: {i}, Loss: {current_loss.item()}")

            optimiser.zero_grad()
            # current_loss.backward()
            # optimiser.step()
            scaler.scale(current_loss).backward()

            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            scaler.step(optimiser)

            # Updates the scale for next iteration.
            scaler.update()
            
            Loss.append(current_loss.item())
            
        if i%100 == 0:
            
            if current_loss.item() < best_loss:
                best_loss = current_loss
                print(f"Saving model at iteration: {i}, Loss: {current_loss}")
                torch.save(model.state_dict(), f"{args.checkpoint_path}/model_{i}.pt")
            
            # print(f" Iteration: {i}, Loss: {current_loss}")



def render_image(model, test_ray_origins, test_ray_directions, test_gt, H, W, args):
    rgb_pred_test = []
    num_rays_per_batch = 4000

    for i in range(H * W // num_rays_per_batch):
        index_1 = i * num_rays_per_batch
        index_2 = (i + 1) * num_rays_per_batch
      
        test_origins, test_directions, gt = generateBatch(test_ray_origins[index_1:index_2], test_ray_directions[index_1:index_2], test_gt[index_1:index_2], args, train=False)      

        # Assuming generateBatch function generates rays for one batch
        pred = render(model, test_origins, test_directions, args)
        # pred = render_rays(model, test_origins, test_directions)
        rgb_pred_test.append(pred)

    rgb_pred_test = torch.cat(rgb_pred_test, dim=0)
    pred_image = rgb_pred_test.view(H, W, 3).cpu().detach().numpy()
    return pred_image

def test(images, poses, camera_info, args):
    
    if args.load_checkpoint:
        # model = TinyNeRFmodel().to(device)
        model = NeRFmodel().to(device)
        # model = NerfModel().to(device)
        # model.load_state_dict(torch.load("Phase2/Output/NeRF-Lego/Checkpoints/model_9900.pt", map_location=device))
        # model.load_state_dict(torch.load("Phase2/Output/checkpoint/model_6900.pt", map_location=device))
        # model.load_state_dict(torch.load("Phase2/Output/checkpoint/TinyNerF-ourData400*400/model_4400.pt", map_location=device))
        model.load_state_dict(torch.load("Phase2/Output/NeRF-Ship/checkpoint/model_6900.pt", map_location=device))

    
    H, W, focal = camera_info
    model.eval()
    
    PSNRs = []
    SSIMs = []
    
    with torch.no_grad():
        test_ray_origins, test_ray_directions, test_gt  = generateRays_and_gt(images, poses, camera_info, args)
        num_images = test_gt.shape[0] // (H*W)
        num_rays_per_image = H*W
        frames = []
        
        for index in range(num_images):
            print(f"Testing on image: {index}")

            if index == 66:
                pred_image = render_image(model, test_ray_origins[num_rays_per_image * index:num_rays_per_image * (index + 1)], test_ray_directions[num_rays_per_image * index:num_rays_per_image * (index + 1)], test_gt, H, W, args)
                gt_image = test_gt[index * num_rays_per_image:(index + 1) * num_rays_per_image].view(H, W, 3).cpu().detach().numpy()
                
                frames.append((255 * pred_image).astype(np.uint8))

                # Calculate PSNR
                psnr = PSNR(gt_image, pred_image)
                PSNRs.append(psnr)
                
                # Calculate SSIM
                ssim = SSIM(gt_image, pred_image)
                SSIMs.append(ssim)
            
            
                if args.plot:
                    # Plotting the original vs predicted images
                    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                    
                    ax[0].imshow(gt_image)
                    ax[0].set_title("Original Test Image")
                    ax[0].axis('off')  # Hide axes ticks
                    
                    ax[1].imshow(pred_image)
                    ax[1].set_title("Predicted Test Image")
                    ax[1].axis('off')  # Hide axes ticks

                    plt.show()

    print(f"Average PSNR: {torch.mean(torch.tensor(PSNRs))}")
    print(f"Average SSIM: {torch.mean(torch.tensor(SSIMs))}")
    
    gif_filename = 'animation_ship.gif'
    imageio.mimsave(gif_filename, frames, fps=30)

    print(f"GIF saved as {gif_filename}")

def PSNR(gt, pred):
    # Convert to tensor
    gt = torch.tensor(gt)
    pred = torch.tensor(pred)
    mse = torch.mean((gt - pred) ** 2)
    return 10 * torch.log10(1.0 / mse)

def SSIM(gt, pred):
    # Convert to tensor
    gt = torch.tensor(gt).permute(2, 0, 1).unsqueeze(0)
    pred = torch.tensor(pred).permute(2, 0, 1).unsqueeze(0)
    # Calculate SSIM
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    return ssim(pred, gt)

def main(args):
    # load data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading data...")
    global test_poses, test_images
    camera_info, images, poses, test_poses, test_images = loadDataset(args.data_path, args.mode, device)
    print("Data loaded")

    if args.mode == 'train':
        print("Start training")
        train(images, poses, camera_info, args)
    elif args.mode == 'test':
        print("Start testing")
        args.load_checkpoint = True
        test(test_images, test_poses, camera_info, args)

def configParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',default="./Phase2/Data/ship/transforms_train.json",help="dataset path")
    parser.add_argument('--mode',default='train',help="train/test/val")
    parser.add_argument('--lrate',default=5e-4,help="training learning rate")
    parser.add_argument('--n_pos_freq',default=10,help="number of positional encoding frequencies for position")
    parser.add_argument('--n_dirc_freq',default=4,help="number of positional encoding frequencies for viewing direction")
    parser.add_argument('--n_rays_batch',default=32*32*4,help="number of rays per batch")
    parser.add_argument('--n_sample',default=192,help="number of sample per ray")
    parser.add_argument('--max_iters',default=10000,help="number of max iterations for training")
    parser.add_argument('--logs_path',default="./logs/",help="logs path")
    parser.add_argument('--checkpoint_path',default="Phase2/Output/checkpoint",help="checkpoints path")
    parser.add_argument('--load_checkpoint',default=True,help="whether to load checkpoint or not")
    parser.add_argument('--save_ckpt_iter',default=1000,help="num of iteration to save checkpoint")
    parser.add_argument('--images_path', default="./image/",help="folder to store images")
    parser.add_argument('--epochs', default=1000, help="number of epochs")
    parser.add_argument('--plot', default=True, help="whether to plot images or not")
    parser.add_argument('--image_size', default=400, help="image size")
    return parser

if __name__ == "__main__":
    parser = configParser()
    args = parser.parse_args()
    main(args)