# /*
#  * @Author: Kunrui Huang 
#  * @Date: 2023-05-06 12:00:16 
#  * @Last Modified by:   Kunrui Huang 
#  * @Last Modified time: 2023-05-06 12:00:16 
#  */
import sys
import os
import struct
import cv2
import numpy as np
import re

type_int = ">i"
type_float = ">f"
type_uint = ">B"


def save_pfm(filename, image, scale=1):
    # the pfm should be .pfm file.
    file = open(filename, "wb")
    color = None

    image = np.flipud(image)

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(
            image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception(
            'Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1],
                                image.shape[0]).encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))

    image.tofile(file)
    file.close()


def load_pfm(file):
    color = None
    width = None
    height = None
    scale = None
    data_type = None
    header = file.readline().decode('UTF-8').rstrip()

    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('UTF-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')
    # scale = float(file.readline().rstrip())
    scale = float((file.readline()).decode('UTF-8').rstrip())
    if scale < 0:  # little-endian
        data_type = '<f'
    else:
        data_type = '>f'  # big-endian
    data_string = file.read()
    data = np.fromstring(data_string, data_type)
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = cv2.flip(data, 0)
    return data


def read_binary_file_depth(mask, file):
    print("Start to process " + file)
    if ("_depth" in file):
        return

    scale = 1000

    with open(file, "rb") as f:
        rows = struct.unpack(type_int, f.read(4))[0]
        cols = struct.unpack(type_int, f.read(4))[0]
        depth = np.zeros((rows, cols), dtype=np.float32)
        depth_png = np.zeros((rows, cols), dtype=np.uint16)

        for i in range(rows):
            for j in range(cols):
                depth_value = struct.unpack(type_float, f.read(4))[0]
                depth[i, j] = depth_value
                depth_png[i, j] = depth_value*scale
        mask = mask > 0
        mask = mask * 1
        depth = depth * mask.astype(np.uint8)
        depth_png = depth_png*mask.astype(np.uint8)
        save_pfm(file + "_depth.pfm", depth)
        cv2.imwrite(file+"_depth.png", depth_png)


def read_binary_file_confidence(file):
    print("Start to process " + file)
    if ("_confidence" in file):
        return

    with open(file, "rb") as f:
        rows = struct.unpack(type_int, f.read(4))[0]
        cols = struct.unpack(type_int, f.read(4))[0]
        confidence = np.zeros((rows, cols), dtype=np.uint8)
        for i in range(rows):
            for j in range(cols):
                confidence_value = struct.unpack('B', f.read(1))[0]
                confidence[i, j] = confidence_value

        confidence_img = confidence * 127
        cv2.imwrite(file + "_confidence.png", confidence_img)

    return confidence


if __name__ == "__main__":
    print("Read the binary depth file.\n")
    confidence_path = sys.argv[1]
    depth_path = sys.argv[2]
    confidence_files = os.listdir(confidence_path)
    confidence_files.sort()
    depth_files = os.listdir(depth_path)
    depth_files.sort()
    current_index = 0
    for file in confidence_files:
        file_abs_path = os.path.join(confidence_path, file)
        confidence_mask = read_binary_file_confidence(file_abs_path)

        depth_file_abs_path = os.path.join(depth_path,
                                           depth_files[current_index])

        read_binary_file_depth(confidence_mask, depth_file_abs_path)

        current_index += 1
