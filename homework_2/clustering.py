import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin, pairwise_distances


def kmeans_find_clusters(x, n_clusters, iter_num=50):
    # 1. Randomly choose clusters

    ind = np.random.permutation(x.shape[0])[:n_clusters]
    centers = x[ind]

    for _ in range(iter_num):
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(x, centers)

        # 2b. Find new centers from means of points
        new_centers = np.array([x[labels == i].mean(0) for i in range(n_clusters)])

        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers

    return centers, labels


class DBSCAN:

    def __init__(self, data, eps, min_points):
        self.data = data
        self.data['visited'] = 0
        self.data['cluster'] = 0
        self.eps = eps
        self.min_points = min_points
        self.clusters = 0

    def cluster(self):
        data = self.data.drop(['visited', 'cluster'], axis=1)
        for row in data.itertuples():
            # print(f'New point {row.Index} {self.data.loc[row.Index].visited} ')
            if self.data.loc[row.Index].visited:
                continue
            self.data.loc[row.Index, 'visited'] = 1
            neighbours = self.find_neighbours(row, data)
            if len(neighbours) < self.min_points:
                self.data.loc[row.Index, 'cluster'] = -1
            else:
                self.clusters += 1
                self.data.loc[row.Index, 'cluster'] = self.clusters
                self.expand_cluster(neighbours)

    def expand_cluster(self, neighbours):
        if np.all(self.data.loc[neighbours.index].visited):
            return
        for row in neighbours.itertuples():
            if not self.data.loc[row.Index].visited:
                self.data.loc[row.Index, 'visited'] = 1
                row_neighbours = self.find_neighbours(row, neighbours)
                if len(row_neighbours) >= self.min_points:
                    # print(f'new neighbours {len(neighbours.append(row_neighbours))}')
                    if set(row_neighbours.index) - set(neighbours.index):
                        self.expand_cluster(neighbours.append(row_neighbours))
            if not self.data.loc[row.Index].cluster:
                self.data.loc[row.Index, 'cluster'] = self.clusters

    def find_neighbours(self, row, data):
        d = np.sqrt(np.sum((data[['X', 'Y']] - [row.X, row.Y]) ** 2, axis=1))
        data['distance'] = d
        return data.loc[data.distance <= self.eps].drop(['distance'], axis=1)


class Objects:
    """ Класс для хранения информации об объектах и кластерах """

    def __init__(self, objects=None):
        """
        Args:
            objects (int or list): либо количество объектов, либо их названия
        """
        # либо сами генерим - номера
        if type(objects) == int:
            self.names = [_ for _ in range(objects)]

        # либо присваиваем те, что пришли
        else:
            self.names = objects

        # начальное количество объектов
        self.number = len(self.names)

        self.__distances = None
        self.__members = None

    @property
    def distances(self):
        """
        Ленивая инициализация расстояний
        """
        if self.__distances is None:
            self.__distances = [0] * self.number
        return self.__distances

    @property
    def members(self):
        """
        Ленивая инициализация количества объектов
        """
        if self.__members is None:
            self.__members = [1] * self.number
        return self.__members


class Cluster:
    """ Abstract """

    def __init__(self, data, objects=None):
        """
        Args:
            data (pd.DataFrame): набор признаков
            objects (list): имена объектов
        """
        self.data = data
        matrix = pairwise_distances(data.values)

        if type(matrix) == list:
            self.matrix = np.array(matrix).astype(float)
        else:
            self.matrix = matrix

        # заполняем диагональные элементы матрицы +бесконечностями, чтобы они не выбирались минимальными
        np.fill_diagonal(self.matrix, np.inf)

        if objects:
            self.objects = Objects(objects)
        else:
            self.objects = Objects(len(matrix))
        self.current_min = None
        self.__linkage = None

    @property
    def linkage(self):
        """
        Newick Standard - красивая строка
        Returns:
            str: linkage
        """
        if self.__linkage is None:
            self.cluster()
        return self.objects.names[0]

    def __repr__(self):
        return self.linkage

    def find_smallest(self, matrix):
        """
        Нахождение минимального индекса в массиве
        Returns:
            tuple: минимальный индекс
        """
        return np.unravel_index(np.argmin(matrix, axis=None), matrix.shape)

    def set_link(self, distance, drop, keep):
        """
        Красивый вывод
        Args:
            keep (int): номер объекта, вместо которого будем писать
            drop (int): номер объекта, который удаляется
            distance (float): расстояние между объектами
        """
        # расстояния равны
        dist1 = dist2 = distance

        # тут получаем имена объектов для записи и расстояния, которые уже имеются до них
        obj_first = self.objects.names[drop]
        obj_first_dist = self.objects.distances[drop]
        obj_second = self.objects.names[keep]
        obj_second_dist = self.objects.distances[keep]

        # переписываем объект
        self.objects.names[keep] = f'({obj_first}: {dist1 - obj_first_dist}, {obj_second}: {dist2 - obj_second_dist})'

        # обновляем его расстояние
        self.objects.distances[keep] = distance

        # удаляем ненужный
        del self.objects.names[drop], self.objects.distances[drop]

    def distance(self):
        """
        Расстояние между объединяемыми объектами
        Returns:
            float: расстояние до каждого
        """
        return self.matrix[self.current_min] / 2

    def new_node(self, *args):
        """
        Обновление узла
        """
        raise NotImplementedError

    def change_matrix(self, distances):
        """
        Обновление матрицы расстояний
        Args:
            distances (float): расстояния до каждого объединяемого узла
        """
        drop, keep = self.current_min

        # объединяем два узла
        self.set_link(distances, drop, keep)

        # обновляем узлы для всех остальных объектов
        idx = set([_ for _ in range(len(self.matrix))]) - set(self.current_min)

        for ind in idx:
            self.matrix[keep, ind] = self.matrix[ind, keep] = self.new_node(ind)

        # удаляем ненужный объект из матрицы
        self.matrix = np.delete(self.matrix, drop, 0)
        self.matrix = np.delete(self.matrix, drop, 1)

        # обновляем кол-во элементов кластера
        self.objects.members[keep] += self.objects.members[drop]
        del self.objects.members[drop]

    def cluster(self):
        """
        Основная функция для кластеризации
        """
        # пока не останется 1 элемент матрицы
        while self.matrix.shape != (1, 1):

            # находим наименьший
            self.current_min = self.find_smallest(self.matrix)

            # получаем расстояния до объединяемых объектов
            distances = self.distance()

            # меняем матрицу расстояний
            self.change_matrix(distances)


class WPGMA(Cluster):
    """ WPGMA """

    def new_node(self, ind):
        """
        Обновление узла
        Args:
            ind (int): номер объекта, расстояние до которого обновляем
        Returns:
            float: новое расстояние
        """
        i, j = self.current_min
        return (self.matrix[i, ind] + self.matrix[j, ind]) / 2
