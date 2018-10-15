import numpy as np
from PIL import Image
from time import ctime


def prepare_image(name: str, height: int, weight: int):
    """Returns matrix of smaller matrixes from image"""
    # read picture to numpy array of size height*weight
    pic = Image.open(name)
    h, w = pic.size[0], pic.size[1]
    pic = np.array(pic.getdata()).reshape(pic.size[0], pic.size[1], 3)
    # generate overlapping row
    row = pic[-height:, :]
    # concatenate overlapping row
    resize_h = h % height if h % height != 0 else height
    pic = np.concatenate((pic[:-resize_h, :], row), axis=0)
    # generate overlapping column
    col = pic[:, -weight:]
    # concatenate overlapping column
    resize_w = w % weight if w % weight != 0 else weight
    pic = np.concatenate((pic[:, :-resize_w], col), axis=1)
    pic = pic.reshape(pic.shape[0] // height, pic.shape[1] // weight, height, weight, 3)
    return pic


mat = prepare_image("ns2.jpg", 10, 10)
N = 10 * 10 * 3
p = N // 4
w1 = np.random.rand(N, p)
print(w1.shape)
w2 = np.random.rand(p, N)
print(w2.shape)
print(ctime())
error_all = 0
k = 0

alpha = 0
while True:
    error_all = 0
    k += 1
    for i in mat:
        for j in i:
            error = 0
            j = j.ravel()
            # print(j.shape)
            j = j.reshape((1, j.shape[0]))
            # print(j.shape)
            y = j.dot(w1)
            # print(y)

            x1 = y.dot(w2)
            # print(x1)

            dx = x1 - j
            # print(dx)

            error = (dx ** 2).sum()
            error_all += error
            # print(1 / (j ** 2).sum())
            # print(1 / (y ** 2).sum())
            # print(w2)
            alpha = 1 / (y ** 2).sum()
            # print(alpha)
            # print(w1 - alpha * j.transpose().dot(dx).dot(w2.transpose()))
            # print(w1)
            w1 -= alpha * j.transpose().dot(dx).dot(w2.transpose())
            # print(w1)
            # print()
            # print(w2)
            w2 -= alpha * y.transpose().dot(dx)
            # print(w2)
            # break
        # break

    # break
    print(k, " ", error_all, " ", ctime())
    if error_all < 5 or k > 5000:
        break
print("ALL")
