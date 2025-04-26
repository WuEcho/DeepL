import numpy as np
import random
import sys
'''
Kmeans算法实现
原文链接：https://blog.csdn.net/qingchedeyongqi/article/details/116806277
'''

class KMeansClusterer:  # k均值聚类
    def __init__(self, ndarray, cluster_num):
        self.ndarray = ndarray
        self.cluster_num = cluster_num
        self.points = self.__pick_start_point(ndarray, cluster_num)

    def cluster(self):
        result = []
        for i in range(self.cluster_num):
            result.append([])  # 初始化结果列表
        for item in self.ndarray:
            distance_min = sys.maxsize
            index = -1
            for i in range(len(self.points)):
                distance = self.__distance(item, self.points[i])
                if distance < distance_min:
                    distance_min = distance
                    index = i  # 找到距离最小的中心点
            result[index] = result[index] + [item.tolist()]  # 将点加入对应的簇
        new_center = []
        for item in result:
            new_center.append(self.__center(item).tolist())  # 计算新的中心点
        # 中心点未改变，说明达到稳态，结束递归
        if (self.points == new_center).all():
            sum = self.__sumdis(result)
            return result, self.points, sum
        self.points = np.array(new_center)
        return self.cluster()

    def __sumdis(self,result):
        #计算总距离和
        sum=0
        for i in range(len(self.points)):
            for j in range(len(result[i])):
                # 计算每个点到中心的距离
                # 这里的result[i]是一个簇
                # result[i][j]是簇中的一个点
                # self.points[i]是对应的中心
                # sum 是所有数据点到其所属簇中心的距离总和，用于衡量聚类的紧密度（总和越小，聚类效果越好）
                sum+=self.__distance(result[i][j],self.points[i])
        return sum

    def __center(self, list):
        # 计算每一列的平均值
        #  axis 表示对哪个维度进行操作
        # 0表示对行进行操作，1表示对列进行操作
        # 这里对每一列进行操作，计算平均值
        # 这里的list是一个二维数组
        # 示例：
        # data = [
        # [1, 2],  # 数据点1
        # [3, 4],  # 数据点2
        # [5, 6]   # 数据点3
        # ]
        ​# axis=0：计算每列的均值（跨所有数据点的同一特征）：
        # np.mean(data, axis=0)  # 结果: [3.0, 4.0]
        # 第一列均值：(1 + 3 + 5) / 3 = 3.0
        # 第二列均值：(2 + 4 + 6) / 3 = 4.0

        # 在K-means中的意义：
        # 簇中心是所有数据点各特征的平均值。例如：
        # 若簇中有3个数据点，每个点有2个特征（如坐标[x, y]），则中心应为：
        # center_x = (x1 + x2 + x3) / 3
        # center_y = (y1 + y2 + y3) / 3
        # 中心 = [center_x, center_y]
        # 这正是axis=0的作用：对每个特征（列）单独求均值
        return np.array(list).mean(axis=0)

    def __distance(self, p1, p2):
        #计算两点间距
        tmp = 0
        for i in range(len(p1)):
            tmp += pow(p1[i] - p2[i], 2)
        return pow(tmp, 0.5)

    def __pick_start_point(self, ndarray, cluster_num):
        if cluster_num < 0 or cluster_num > ndarray.shape[0]:
            raise Exception("簇数设置有误")
        # 取点的下标
        indexes = random.sample(np.arange(0, ndarray.shape[0], step=1).tolist(), cluster_num)
        points = []
        for index in indexes:
            points.append(ndarray[index].tolist())
        return np.array(points)

x = np.random.rand(100, 8)
kmeans = KMeansClusterer(x, 10)
result, centers, distances = kmeans.cluster()
print(result)
print(centers)
print(distances)

