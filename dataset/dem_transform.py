import numpy as np
import math
from PIL import Image
import os


#####在原栅格图像周围加一圈并返回
def AddRound(npgrid):
    ny, nx = npgrid.shape  # ny:行数，nx:列数
    zbc = np.zeros((ny + 2, nx + 2))
    zbc[1:-1, 1:-1] = npgrid
    # 四边
    zbc[0, 1:-1] = npgrid[0, :]
    zbc[-1, 1:-1] = npgrid[-1, :]
    zbc[1:-1, 0] = npgrid[:, 0]
    zbc[1:-1, -1] = npgrid[:, -1]
    # 角点
    zbc[0, 0] = npgrid[0, 0]
    zbc[0, -1] = npgrid[0, -1]
    zbc[-1, 0] = npgrid[-1, 0]
    zbc[-1, -1] = npgrid[-1, 0]
    return zbc


#####计算xy方向的梯度
def Cacdxdy(npgrid, sizex, sizey):
    zbc = AddRound(npgrid)
    dx = ((zbc[1:-1, :-2]) - (zbc[1:-1, 2:])) / sizex / 2
    dy = ((zbc[2:, 1:-1]) - (zbc[:-2, 1:-1])) / sizey / 2
    dx = dx[1:-1, 1:-1]
    dy = dy[1:-1, 1:-1]
    np.savetxt("dxdy.csv", dx, delimiter=",")
    return dx, dy


####计算坡度\坡向
def CacSlopAsp(dx, dy):
    slope = (np.arctan(np.sqrt(dx * dx + dy * dy))) * 57.29578  # 转换成°
    # slope = slope[1:-1, 1:-1]
    # 坡向
    a = np.zeros([dx.shape[0], dx.shape[1]]).astype(np.float32)
    for i in range(dx.shape[0]):
        for j in range(dx.shape[1]):
            x = float(dx[i, j])
            y = float(dy[i, j])
            if (x == 0.) & (y == 0.):
                a[i, j] = -1
            elif x == 0.:
                if y > 0.:
                    a[i, j] = 0.
                else:
                    a[i, j] = 180.
            elif y == 0.:
                if x > 0:
                    a[i, j] = 90.
                else:
                    a[i, j] = 270.
            else:
                a[i, j] = float(math.atan(y / x)) * 57.29578
                if a[i, j] < 0.:
                    a[i, j] = 90. - a[i, j]
                elif a[i, j] > 90.:
                    a[i, j] = 450. - a[i, j]
                else:
                    a[i, j] = 90. - a[i, j]
    return slope, a

import random
def trans_dem(dem):
    dem = AddRound(dem)
    dx, dy = Cacdxdy(dem, 1, 1)
    slope, arf = CacSlopAsp(dx, dy)
    if random.random()>0.5:
        return slope
    else:
        return arf

from osgeo import gdal
def trans_dem_gdal(dem):
    gdal.DontUseExceptions()
    mem_driver = gdal.GetDriverByName('MEM')
    dem_data = mem_driver.Create('', dem.shape[0], dem.shape[1], 1, gdal.GDT_Float32)
    dem_data.GetRasterBand(1).WriteArray(dem)
    Options = gdal.DEMProcessingOptions(format='MEM', computeEdges=True, zeroForFlat=True)
    slope = gdal.DEMProcessing('', dem_data, 'slope', options=Options)
    # aspect = gdal.DEMProcessing('', dem_data, 'aspect', options=Options)
    Roughness = gdal.DEMProcessing('', dem_data, "Roughness", options=Options)
    TPI = gdal.DEMProcessing('', dem_data, "TPI", options=Options)

    slope = slope.ReadAsArray()
    # aspect = aspect.ReadAsArray()
    Roughness = Roughness.ReadAsArray()
    TPI = TPI.ReadAsArray()

    slope = 255-((slope - np.min(slope)) / (np.max(slope) - np.min(slope) + 1e-5) * 255).astype(np.uint8)
    # aspect = ((aspect - np.min(aspect)) / (np.max(aspect) - np.min(aspect)) * 255).astype(np.uint8)
    Roughness = 255-((Roughness - np.min(Roughness)) / (np.max(Roughness) - np.min(Roughness) + 1e-5) * 255).astype(np.uint8)
    TPI = ((TPI - np.min(TPI)) / (np.max(TPI) - np.min(TPI) + 1e-5) * 255).astype(np.uint8)

    a = random.random()

    # return
    if a < 0.3:
        return slope
    elif a >= 0.3 and a < 0.6:
        return Roughness
    else:
        return TPI


def PCA_my(image, k):
    # 将图像展平为二维数组（每列是一个像素，每行是一个通道）
    h, w, c = image.shape
    data = image.reshape(-1, c).astype(np.float32)  # 转换为列向量

    # 计算均值和标准差
    mean = np.mean(data, axis=0)
    data_centered = data - mean  # 中心化数据

    # 计算协方差矩阵
    covariance_matrix = np.cov(data_centered, rowvar=False)

    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # 排序特征值和特征向量
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # 选择前k个特征向量（例如：k=2）
    selected_eigenvectors = eigenvectors[:, :k]

    # 将数据投影到新的特征空间
    projected_data = np.dot(data_centered, selected_eigenvectors).reshape(h, w, c)
    projected_data = ((projected_data - np.min(projected_data)) / (
                np.max(projected_data) - np.min(projected_data)) * 255).astype(
        np.uint8)

    return projected_data

def NDVI(image):
    red_band = image[:, :, 0].astype(np.float32)  # 红光波段
    nir_band = image[:, :, 3].astype(np.float32)  # 近红外波段
    # 计算NDVI
    ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-5)  # 加小值避免除零
    ndvi = ((ndvi - np.min(ndvi)) / (np.max(ndvi) - np.min(ndvi) + 1e-5) * 255).astype(np.uint8)
    return 255-ndvi

# import os
# os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
# import cv2

from sklearn.decomposition import PCA
def trans_img_RS(img):
    # img: 4bands
    pca = PCA(n_components=4)
    # 进行 PCA 转换
    img_pca = pca.fit_transform(img.reshape(-1, 4))
    # 将结果转换回图像格式
    img_pca = img_pca.reshape(img.shape[0], img.shape[1], 4)[:, :, 1]
    img_pca = 255-((img_pca - np.min(img_pca)) / (np.max(img_pca) - np.min(img_pca) + 1e-5) * 255).astype(np.uint8)

    img_pca = np.stack([img_pca] * 3, axis=-1)
    ndvi = NDVI(img)
    ndvi = np.stack([ndvi] * 3, axis=-1)

    a = random.random()
    if a < 0.5:
        return img_pca
    else:
        return ndvi



if __name__ == '__main__':
    print(1)






