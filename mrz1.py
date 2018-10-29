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
    pic = pic.reshape(
        pic.shape[0] // height, pic.shape[1] // weight, 1, height * weight * 3
    )
    return pic


def restore_image(
    name: str,
    mat: np.ndarray,
    height: int,
    weight: int,
    output_hight: int,
    output_weight: int,
):
    """Restore image from an array of rectangles and save it to bmp imgae"""
    mat = 255 * (mat + 1) / 2
    mat[mat < 0] = 0
    mat[mat > 255] = 255
    # reshape matrix to imaging array with duplicated columns and rows
    resize_h = output_hight % height if output_hight % height != 0 else height
    resize_w = output_weight % weight if output_weight % weight != 0 else weight
    mat = mat.reshape(
        output_hight + height - resize_h, output_weight + weight - resize_w, 3
    )
    # delete extra col from matrix, between -2 * weight and -weight
    # mat = np.concatenate((mat[:, : -2 * weight + 1], mat[:, -weight:]), axis=1)
    # delete extra col from matrix, between -2 * height and -height
    # mat = np.concatenate((mat[: -2 * height + 1, :], mat[-height:, :]), axis=0)
    img = Image.fromarray(mat.astype("uint8"), "RGB")
    img.save(name, format="BMP")


def start():
    """Take start arguments from user and runs learning alghoritm"""
    n = int(input("Height of rectangle:\n"))
    m = int(input("Weight of rectangle:\n"))
    p = int(input("Number of neurons on second layer:\n"))
    alpha = float(input("Alpha, enter 0 to auto select:\n"))

    if alpha == 0.0:
        alpha = 0.001877
    mat = prepare_image("256px-Lenna.png", n, m)
    # number of neurons in first and last layers
    N = n * m * 3
    execute(mat, alpha, N, p, n, m, 256, 256)


def execute(
    mat: np.ndarray,
    alpha: float,
    N: int,
    p: int,
    height: int,
    weight: int,
    output_height: int,
    output_weight: int,
):
    """This function provides learning network by rectangles of image"""
    w1 = np.random.rand(N, p) * 2 - 1
    w2 = np.random.rand(p, N) * 2 - 1
    print(ctime())
    error_all = 0
    mat = (2 * mat / 255) - 1
    k = 0
    while True:
        error_all = 0
        k += 1
        x1 = np.ndarray((1, 0), dtype=float)
        time1 = time()
        for i in mat:
            for j in i:
                # error = 0
                # j = j.ravel()
                # j = j.reshape((1, j.shape[0]))
                y = np.matmul(j, w1)
                x1 = np.matmul(y, w2)
                dx = x1 - j
                # error = (dx * dx).sum()
                # error_all += error
                w1 -= alpha * np.matmul(np.matmul(j.transpose(), dx), w2.transpose())
                w2 -= alpha * np.matmul(y.transpose(), dx)
                # break
            # break
        for i in mat:
            for j in i:
                error = 0
                # j = j.ravel()
                # j = j.reshape((1, j.shape[0]))
                y = np.matmul(j, w1)
                x1 = np.matmul(y, w2)
                dx = x1 - j
                error = (dx * dx).sum()
                error_all += error
        time2 = time()
        print("time for iteration ", time2 - time1)
        # break
        print(k, " ", error_all)
        if k % 10 == 0:
            res = []
            for i in range(len(mat)):
                row = []
                for j in range(len(mat[i])):
                    row.append(np.matmul(np.matmul(mat[i, j], w1), w2))
                res.append(row)
            res = np.array(res)
            restore_image(
                "output" + str(k), res, height, weight, output_height, output_weight
            )
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
            restore_image("output", res, height, weight, output_height, output_weight)
            break
    print("ALL")


if __name__ == "__main__":
    start()
