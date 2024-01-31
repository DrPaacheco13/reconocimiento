"""
 Copyright (c) 2018-2023 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import numpy as np

from model_api.models.utils import resize_image


# import numpy as np

def crop(frame, roi):
    margin = 30  # ajusta el valor del margen según tus necesidades

    if hasattr(roi, 'position'):
        p1 = roi.position.astype(int) - margin
        p1 = np.clip(p1, [0, 0], [frame.shape[1], frame.shape[0]])
        p2 = (roi.position + roi.size).astype(int) + margin
        p2 = np.clip(p2, [0, 0], [frame.shape[1], frame.shape[0]])
    else:
        p1 = max(np.array(roi[0]).astype(int) - margin, [0, 0])
        p1 = min(p1, [frame.shape[1], frame.shape[0]])

        p2 = max((np.array(roi[0]) + np.array(roi[1])).astype(int) + margin, [0, 0])
        p2 = min(p2, [frame.shape[1], frame.shape[0]])

    return frame[p1[1]:p2[1], p1[0]:p2[0]]



def cut_rois(frame, rois):
    return [crop(frame, roi) for roi in rois]


def resize_input(image, target_shape, nchw_layout):
    if nchw_layout:
        _, _, h, w = target_shape
    else:
        _, h, w, _ = target_shape
    # print('----------resize-----------------')
    # print(w)
    # print(h)
    # print(image)
    # print('----------resize-----------------')
    resized_image = resize_image(image, (w, h))
    if nchw_layout:
        resized_image = resized_image.transpose((2, 0, 1)) # HWC->CHW
    resized_image = resized_image.reshape(target_shape)
    return resized_image
