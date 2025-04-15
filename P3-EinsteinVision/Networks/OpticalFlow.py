import numpy as np
import cv2
import ptlflow
from ptlflow.utils import flow_utils
from ptlflow.utils.io_adapter import IOAdapter

model = ptlflow.get_model("rpknet", ckpt_path="kitti").cuda().half()



def get_optical_flow(image1, image2):
    """
    Calculate the optical flow between two images using the PWC-Net model.

    Parameters
    ----------
    image1 : np.ndarray
        The first image.
    image2 : np.ndarray
        The second image.

    Returns
    -------
    np.ndarray
        The optical flow between the two images of the shape of (H, W, 2) where the channel 0 is the x-displacement
        and channel 1 is the y-displacement.

    """
    # Convert images to float32 and normalize to [0, 1]
    h, w = image1.shape[:2]
    imgs = np.stack([image1, image2], axis=0)

    # inputs is a dict {'images': torch.Tensor}
    # The tensor is 5D with a shape BNCHW. In this case, it will have the shape:
    # (1, 2, 3, H, W)

    io_adapter = IOAdapter(model,input_size=(h,w), target_size=(h//2, w//2), cuda=True, fp16=True)
    inputs = io_adapter.prepare_inputs(imgs)

    # Forward the inputs through the model
    predictions = model(inputs)

    # The output is a dict with possibly several keys,
    # but it should always store the optical flow prediction in a key called 'flows'.
    flows = predictions['flows']

    # flows will be a 5D tensor BNCHW.
    # This example should print a shape (1, 1, 2, H, W).
    # We need to permute the tensor to get the shape (H, W, 2)
    flow = flows.squeeze().permute(1, 2, 0)
    flow = flow.detach().cpu().numpy()

    return flow



def visualize(flow: np.array, sz: tuple[int, int] = (1280, 960)) -> np.array:
    # Create an RGB representation of the flow to show it on the screen
    # flow is a numpy array with shape (H, W, 2)
    flow_rgb: np.array = flow_utils.flow_to_rgb(flow)
    # Make it a numpy array with HWC shape
    # flow_rgb = np.permute_dims(flow_rgb, (1, 2, 0))
        # OpenCV uses BGR format
    flow_bgr_npy = cv2.cvtColor(flow_rgb, cv2.COLOR_RGB2BGR)
    flow_bgr_npy = cv2.resize(flow_bgr_npy, sz, fx=0.5, fy=0.5)

    return flow_bgr_npy

