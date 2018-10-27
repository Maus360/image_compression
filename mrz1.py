import numpy as np
from PIL import Image
from time import ctime, time


def prepare_image(name: str, height: int, weight: int) -> np.ndarray:
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


def restore_image(name: str, mat: np.ndarray, height: int, weight: int):
    mat = 255 * (mat + 1) / 2
    print(mat.shape)
    mat = mat.reshape(256, 256, 3)
    print(mat.shape)
    img = Image.fromarray(mat.astype("uint8"), "RGB")
    img.save(name, format="BMP")


mat = prepare_image("256px-Lenna.png", 4, 4)
N = 4 * 4 * 3
p = 15
w1 = np.random.rand(N, p)
print(w1.shape)
w2 = np.random.rand(p, N)
print(ctime())
error_all = 0
k = 0
times = []
alpha = 0
while True:
    error_all = 0
    k += 1
    x1 = np.ndarray((1, 0), dtype=float)
    time1 = time()
    for i in mat:
        for j in i:

            error = 0

            j = j.ravel()
            # time2 = time()
            # print("j.ravel times ", time2 - time1)
            # print(j.shape)
            # time1 = time()
            j = j.reshape((1, j.shape[0]))
            # time2 = time()
            # print("j.reshape times ", time2 - time1)
            # time1 = time()
            x = (2 * j / 255) - 1
            # time2 = time()
            # print("compute x times ", time2 - time1)
            # print(j.shape)
            # time1 = time()
            y = np.dot(x, w1)
            # time2 = time()
            # print("x.dot(w1) ", time2 - time1)
            # print(y)
            # time1 = time()
            x1 = np.dot(y, w2)
            # time2 = time()
            # print("y.dot(w2) times ", time2 - time1)
            # print(x1)
            # time1 = time()
            dx = x1 - x
            # time2 = time()
            # print("dx times ", time2 - time1)
            # print(dx)
            # time1 = time()
            error = (dx ** 2).sum()
            # time2 = time()
            # print("error times ", time2 - time1)
            # time1 = time()
            error_all += error
            # time2 = time()
            # print("error all times ", time2 - time1)
            # print(1 / (j ** 2).sum())
            # print(1 / (y ** 2).sum())
            # print(w2)
            alpha = 0.000977
            # print(alpha)
            # print(w1 - alpha * j.transpose().dot(dx).dot(w2.transpose()))
            # print(w1)
            # time1 = time()
            w1 -= alpha * np.dot(np.dot(x.transpose(), dx), w2.transpose())
            # time2 = time()
            # print("compute w1 times ", time2 - time1)
            # w1 /= (w1 ** 2).sum() ** 0.5
            # alpha_ = 1 / (x1 ** 2).sum()
            # print(w1)
            # print()
            # print(w2)
            # time1 = time()
            w2 -= alpha * np.dot(y.transpose(), dx)
            # time2 = time()
            # print("compute w2 times ", time2 - time1)
            # w2 /= (w2 ** 2).sum() ** 0.5
            # print(w2)
            # break
        # break
    time2 = time()
    print("time for iteration ", time2 - time1)
    # break
    print(k, " ", error_all)
    if k % 1 == 0:
        res = []
        for i in range(len(mat)):
            row = []
            for j in range(len(mat[i])):
                y = (
                    (
                        (
                            2
                            * mat[i, j]
                            .ravel()
                            .reshape(1, mat.shape[2] * mat.shape[3] * mat.shape[4])
                            / 255
                        )
                        - 1
                    )
                    .dot(w1)
                    .dot(w2)
                )
                row.append(y)
            res.append(row)
        res = np.array(res)
        restore_image("output1" + str(k), res, 4, 4)
    if error_all < 100:
        res = []
        for i in range(len(mat)):
            row = []
            for j in range(len(mat[i])):
                y = (
                    (
                        (
                            2
                            * mat[i, j]
                            .ravel()
                            .reshape(1, mat.shape[2] * mat.shape[3] * mat.shape[4])
                            / 255
                        )
                        - 1
                    )
                    .dot(w1)
                    .dot(w2)
                )
                row.append(y)
            res.append(row)
        res = np.array(res)
        restore_image("output1", res, 10, 10)
        break
print("ALL")
