import cv2
import numpy as np
# 画直线 矩形 圆 椭圆 多边形  添加文本
import torch
import torch.nn.functional as F
import torch.nn as nn



def paint():
    #img = cv2.imread('confu2.jpg')
    img = np.zeros((512,512,3),np.uint8)
    cv2.namedWindow('littlejun')

    # 画直线
    #        画布 起点 末点      颜色bgr  thickness  抗锯齿类型
    cv2.line(img,(1,1),(512,512),(255,0,0),1,cv2.LINE_AA)
    # 长方形
    #             画布 左上角 右下角     颜色    线大  线的类型
    cv2.rectangle(img,(412,0),(512,100),(0,255,0),2,cv2.LINE_AA)
    # 圆
    #            画布  圆心  半径  颜色    线大  线的类型
    cv2.circle(img,(200,200),2,(0,0,255),2,cv2.LINE_AA)
    cv2.imshow('littlejun', img)
    cv2.waitKey(0)
    # 椭圆
    #           画布   椭圆中心 （长半轴，短半轴）x轴偏离水平方向的位置（正数顺时针，负数逆时针），颜色 线大（负数为填充模式）线的类型
    cv2.ellipse(img,(100,462),(100,50),-45,0,180,(255,255,255),-1,cv2.LINE_AA)

    pts = np.array([[0,0],[30,40],[100,200],[50,30]],np.int32)
    #改变形状的原因是  将这些点设为一个形状为ROWSx1x2 的数组，其中ROWS是顶点的数目
    pts = pts.reshape((-1,1,2))  # 变成4层 1行两列的三维矩阵 -1 代表由后面的维数算出确定
    #                 以列表的形式窜入 第三个参数为真是图是封闭的，否则只是把点连起来的折线
    cv2.polylines(img,[pts],1,(0,255,255))

    # 画多条直线  先创建直线数组，然后 传过去
    line1 = np.array([[150,250],[350,450]],np.int32)
    line2 = np.array([[10,20],[30,40]],np.int32)
    line3 = np.array([[350,400],[345,480]],np.int32)

    #  注意p3 设为假
    cv2.polylines(img,[line1,line2,line3],0,(255,255,255))

    #添加文本
    #           画布  文字                            左上角位置   字体               字体大小1比较合适 0.5也是可以的  颜色  线条粗细
    cv2.putText(img,'littlejun size 1 hershey simplex',(5,300),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1)
    cv2.putText(img,'littlejun size 1 comlex',(5,330),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
    cv2.putText(img,'littlejun size 2 plain',(5,360),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),1)
    # 都没有
    # cv2.putText(img,'littlejun size 1 duplex',(5,400),cv2.FONT_FONT_HERSHEY_DUPLEX,1,(255,255,255),1)
    # cv2.putText(img,'littlejun size 1 triplex',(5,450),cv2.FONT_FONT_HERSHEY_TRIPLEX,1,(255,255,255),1)

    cv2.imshow('littlejun',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# shape[h, w]
"""
[[0.00136172 0.00519445 0.01471566 0.03096055 0.04837557 0.05613476 0.04837557 0.03096055 0.01471566 0.00519445 0.00136172]
 [0.01180764 0.04504176 0.12760152 0.26846323 0.41947123 0.48675226 0.41947123 0.26846323 0.12760152 0.04504176 0.01180764]
 [0.02425801 0.09253528 0.26214881 0.55153977 0.86177563 1.         0.86177563 0.55153977 0.26214881 0.09253528 0.02425801]
 [0.01180764 0.04504176 0.12760152 0.26846323 0.41947123 0.48675226 0.41947123 0.26846323 0.12760152 0.04504176 0.01180764]
 [0.00136172 0.00519445 0.01471566 0.03096055 0.04837557 0.05613476 0.04837557 0.03096055 0.01471566 0.00519445 0.00136172]]

"""
def ellipseGaussian2D(shape, sigma):
    h, w = [(s-1)/2. for s in shape]
    y, x = np.ogrid[-h:h+1, -w:w+1]
    value = np.exp(-(x*x)/(2*sigma[1]*sigma[1])-(y*y)/(2*sigma[0]*sigma[0]))
    value[value<np.finfo(value.dtype).eps * value.max()] = 0
    return value

def Gausssian2D():
    kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
              [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
              [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
              [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
              [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]
    x = torch.Tensor(kernel)
    print(x.sum())
    exit(1)

def GaussianKernel2D(size=3, sigma=None):
    if sigma is None:
        sigma = size / 6.
    assert size % 2 == 1, "only support odd kernel size"
    h = w = (size-1)/2
    y, x = np.ogrid[-h:h+1, -w:w+1]
    value = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    value[value < np.finfo(value.dtype).eps * value.max()] = 0
    # [h,w]
    weight = torch.FloatTensor(value)
    weight = weight / weight.sum()
    return weight


class GaussianConv(nn.Module):
    def __init__(self, in_planes, kernel_size=3, sigma=None):
        super(GaussianConv, self).__init__()
        # [kernel_size, kernel_size]
        kernel_weight = GaussianKernel2D(kernel_size, sigma)
        kernel_weight = kernel_weight.unsqueeze(0).unsqueeze(0)
        # repeat for each in_planes
        kernel_weight = kernel_weight.repeat(in_planes, 1, 1, 1)
        print(kernel_weight.dtype)
        self.in_planes = in_planes
        self.weight = nn.Parameter(data=kernel_weight, requires_grad=False)
        self.padding = kernel_size // 2

    def __call__(self, x):
        x = F.conv2d(x, self.weight, padding=self.padding, groups=self.in_planes)
        return x

def extract_bbox_from_map(boolen_map):
    assert boolen_map.ndim == 2, 'Invalid input shape'
    #[w,h]
    rows = np.any(boolen_map, axis=1)
    cols = np.any(boolen_map, axis=0)
    if rows.max() == False or cols.max() == False:
        return 0, 0, 0, 0
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return xmin, ymin, xmax, ymax
"""
距离变换函数 cv2.distanceTransform（）计算二值图像内任意点到最近背景点的距离。一般情况下，该函数计算的是图像内非零值像素点到最近的零值像素点的距离，即计算二值图像中所有像素点距离其最近的值为 0 的像素点的距离。
当然，如果像素点本身的值为 0，则这个距离也为 0。
距离变换函数 cv2.distanceTransform（）的计算结果反映了各个像素与背景（值为 0 的像素点）的距离关系。通常情况下：
如果前景对象的中心（质心）距离值为 0 的像素点距离较远，会得到一个较大的值。
如果前景对象的边缘距离值为 0 的像素点较近，会得到一个较小的值。
如果对上述计算结果进行阈值化，就可以得到图像内子图的中心、骨架等信息。距离变换函数 cv2.distanceTransform（）可以用于计算对象的中心，还能细化轮廓、获取图像前景等，有多种功能。
"""
def get_bbox(img, g_ths):
    '''
    :param img: single channel heatmap, np.ndarray
    :param g_ths: list of binarization threshold, [th_1, th_2, ..., th_n]
    :return: bboxes [N, (x, y, w, h)]
    '''
    H, W = img.shape
    bboxes = []
    for th in g_ths:                 # src, thresh, maxval, type
        # th, binary = cv2.threshold (源图片, 阈值, 填充色, 阈值类型), binary 二值化的图片， 大于阈值填充伪指定值
        _, binary = cv2.threshold(img, th, 255, cv2.THRESH_BINARY)
        binary = binary.astype(np.uint8)
        # Distance Transform
        binary = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        binary[binary > 255.0] = 255.0
        binary = binary.astype(np.uint8)
        # 距离最近背景大于3的都当作是前景
        _, binary = cv2.threshold(binary, 3, 255, cv2.THRESH_BINARY)

        contours, hie = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            bbox = cv2.boundingRect(contour)
            x, y, w, h = bbox
            x = min(max(x, 0), W-5)
            y = min(max(y, 0), H-5)
            w = min(max(w, 0), W-x-5)
            h = min(max(h, 0), H-y-5)
            bboxes.append([x, y, w, h])

    return bboxes
"""
contours, hierarchys = cv2.findContours(image, mode, method[, contours[, hierarchy[, offset ]]]) 
mode 可选类型：
cv2.RETR_EXTERNAL     表示只检测外轮廓
cv2.RETR_LIST         检测的轮廓不建立等级关系
cv2.RETR_CCOMP        建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。
cv2.RETR_TREE          建立一个等级树结构的轮廓。

第三个参数method为轮廓的近似办法
cv2.CHAIN_APPROX_NONE 存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1-x2），abs（y2-y1））==1
cv2.CHAIN_APPROX_SIMPLE 压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息

返回值：
contours: list[list[point], 是一个二维列表,contours[i] 表示第i个轮廓所存储的点
hierarchys: list[list[4]], 记录每一个轮廓相关的4种轮廓的索引 
其中的元素个数和轮廓个数相同，每个轮廓contours[i]对应4个hierarchy元素hierarchy[i][0]~hierarchy[i][3]，
分别表示后一个轮廓、前一个轮廓、父轮廓、内嵌轮廓的索引编号，如果没有对应项，则该值为负数。
"""
# 单目标
def test_boxes():
    # box [1,0, 3,2]
    x = [[0,0,1,0],
         [0,1,1,1],
         [0,1,1,0],
         [0,0,0,0]]
    x = np.array(x)
    # 判断每一行是否有1,在dim=1 维度找
    rows = np.any(x, axis=1)
    # 判断每一列是否有1， 在dim=0 维度找
    cols = np.any(x, axis=0)
    # print(rows)
    # print(cols)
    # [True  True  True False]
    # [False  True  True  True]
    ind = np.where(rows)[0] # 返回的是一个tuple 索引,所以要用[0] 获取
    ymin, ymax = ind[[0,-1]] # [0,-1] 获取起止行
    ind = np.where(cols)[0]
    xmin, xmax = ind[[0,-1]] # [0,-1] 获取起止列
    print(xmin, ymin, xmax, ymax)

def test_KLloss():
    loss_fn = nn.KLDivLoss(reduction='none')
    x = torch.rand(2, 4)
    y = torch.rand(2, 4)
    x = F.softmax(x, dim=-1)
    y = F.softmax(y, dim=-1)
    loss = loss_fn(x, y)
    print(loss)

def test_neg_ind():
    topk = 128
    neg_k = 32
    bs = 4
    neg_indices = torch.zeros(bs, topk)
    cand_ind = np.arange(topk)
    # [neg_k]
    neg_ind = np.random.choice(cand_ind, neg_k, replace=False)
    neg_ind
    print("neg_ind", neg_ind.shape)
    # [bs, neg_k]
    neg_ind = neg_ind[None].repeat(bs, axis=0)
    print("bs_neg_ind", neg_ind.shape)
    print(neg_ind)
    print("before neg_indcies", neg_indices.size())
    neg_indices = neg_indices[neg_ind].reshape(bs, -1)
    print("after neg_indices", neg_indices.size())

def test_tau():
    x = torch.randn(2, 10)
    ex = torch.exp(x)


if __name__ == "__main__":
    # paint()
    # value = ellipseGaussian2D([5,11], sigma=[5/6, 11/6])
    # print(value)
    # Gausssian2D()
    # GaussianKernel2D(3)
    # model = GaussianConv(3)
    # x = torch.rand(1,3, 5, 5)
    # # print(x.dtype) # torch.float32
    # print(x)
    # res = model(x)
    # print(res.size())
    # print(res)
    # test_boxes()
    # test_KLloss()
    test_neg_ind()