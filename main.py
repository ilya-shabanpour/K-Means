from PIL import Image
import numpy as np
from sklearn.cluster import KMeans


def vectorize(image):
    data = np.asarray(image)
    arr = []
    arr = np.array(arr)
    list_ones = []
    for i in range(34):
        for j in range(33):
            row_start = 16*i
            row_end = 16*(i+1)
            col_start = 16*j
            col_end = 16*(j+1)
            arr = data[row_start:row_end, col_start:col_end]
            list_ones.append(arr.flatten())
    return list_ones


if __name__ == '__main__':
    img = Image.open("usps_1.jpg")
    list_ones = vectorize(img)
    img = Image.open("usps_2.jpg")
    list_twos = vectorize(img)
    img = Image.open("usps_3.jpg")
    list_threes = vectorize(img)
    img = Image.open("usps_4.jpg")
    list_fours = vectorize(img)
    img = Image.open("usps_5.jpg")
    list_fives = vectorize(img)

    list = list_ones + list_twos + list_threes + list_fours + list_fives
    list = np.array(list)
    k = 5
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
    kmeans = kmeans.fit(list)
    # print(kmeans.cluster_centers_)
    clus_center_list = kmeans.cluster_centers_
    c = 0
    center_matrix = []
    for i in range(k):
        temp_list2 = []
        for j in range(16):
            temp_list = []
            for k in range(16):
                temp_list.append(clus_center_list[i][c])
                c += 1
            temp_list2.append(temp_list)
        c = 0
        center_matrix.append(temp_list2)
    center_matrix = np.array(center_matrix)
    img = Image.fromarray(center_matrix[3])
    img.show()
