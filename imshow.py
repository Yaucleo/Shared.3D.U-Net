import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import scipy.ndimage
import nrrd
from PIL import Image
import matplotlib.pyplot as plt
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# 三维图片可视化函数
# image 输入图片
# threshold 像素阈值，像素值大于该值的像素会显示，否则不显示
# 参考链接：https://zhuanlan.zhihu.com/p/59413289
def plot_3d(image, threshold=-300):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)
    # verts, faces = measure.marching_cubes(p, threshold)
    verts, faces = measure.marching_cubes_classic(p, threshold)
    # marching_cubes_classic/
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    plt.show()


# 使用测试
if __name__ == '__main__':
    nrrd_filename = 'D:/CJY/myData/testSet/000_laendo.nrrd'
    nrrd_data, nrrd_options = nrrd.read(nrrd_filename)
    nrrd_image = Image.fromarray(nrrd_data[:, :, 44] * 1.5)
    plot_3d(nrrd_data, 100)
