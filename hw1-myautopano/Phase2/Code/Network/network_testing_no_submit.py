import torch
import TensorDLT
import numpy
import cv2

def test_tensorDLT():
    test_L = torch.tensor([
        [
            [-23, -18, -29, 29, -5, -22, 5, 32],
            [-213, -181, -219, 219, -15, -212, 15, 312],
            [-223, -281, -229, 229, -25, -222, 25, 322],
            [-233, -381, -239, 239, -35, -232, 35, 322],
        ],
        [
            [-1, -8, -91, 9, -65, -4, -5, 37],
            [-233, -118, -229, 291, -513, -21, 52, -20],
            [-243, -128, -239, 292, -523, -22, 62, -30],
            [-253, -138, -249, 293, -533, -23, 72, -40]
        ]
    ], dtype=torch.float64)
    actual_H4pt_list = [
        [
            [0, 0, 128, 0, 0, 128, 128, 128],
            [-213, -181, -219, 219, -15, -212, 15, 312],
        ],
        [
            [-23, -18, 99, 29, -5, 106, 133, 160],
            [-233, -118, -229, 291, -513, -21, 52, -20],
        ]

    ]

    import numpy as np
    H4pt_src = np.array(actual_H4pt_list[0][0]).reshape(4, 2)
    print(f'src:{H4pt_src}')
    H4pt_dst = np.array(actual_H4pt_list[1][0]).reshape(4, 2)
    print(f'dst:{H4pt_dst}')

    import cv2
    H_cv2, _ = cv2.findHomography(H4pt_src, H4pt_dst)

    H4pt_torch = torch.tensor(actual_H4pt_list, dtype=torch.float64)

    # test_img = torch.zeros(size=(2,128,128))
    H_torch = TensorDLT.forward(H4pt_torch[0], H4pt_torch[1])
    print(H_cv2)
    print(H_torch[0])



if __name__ == '__main__':
    test_tensorDLT()
