#!/usr/bin/python

import numpy as np
import os
from PIL import Image

import sys
sys.path.insert(0,'util/quadric')
from voxelfuse.voxel_model import VoxelModel
from voxelfuse.mesh import Mesh
import argparse

def main(argv):
    parser = argparse.ArgumentParser(description='Personal information')
    parser.add_argument('--filename', dest='filename', type=str, help='Name of depth image')
    parser.add_argument('--maxheight', dest='maxheight', type=str, help='Maximum height in relation to image width/height unit')
    args = parser.parse_args()
    img = read_img(args.filename)
    voxels = create_voxels(img, max_height=int(args.maxheight))
    render_mesh(voxels, args.filename)

def read_img(filename):
    img = Image.open(filename).convert('L')
    # The transpose is a bit hacky and only necessary for the OpenFOAM use case
    img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    img = np.array(img)
    return img

def create_voxels(img, max_height=3):
    voxels = np.zeros([img.shape[0], img.shape[1], max_height])
    # Z = np.floor(Z).astype(int)
    img = img/255
    img = img * max_height
    img = img.astype(int)
    print(img)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            for z in range(img[x,y]):
                voxels[x, y, z] = 1

    return voxels

def render_mesh(img, filename):
    model = VoxelModel(img)  # , generateMaterials(4)  4 is aluminium.
    mesh = Mesh.fromVoxelModel(model)
    split = os.path.splitext(filename)
    fname = split[0]
    mesh.export(fname + '.stl')


if __name__ == "__main__":
    main(sys.argv[1:])