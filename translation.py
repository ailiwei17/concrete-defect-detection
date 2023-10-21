import numpy as np


def translate(obj_x, obj_y, obj_z):
    z_offset = -0.29
    x_offset = -0.070
    z = obj_z + z_offset
    x = obj_y + x_offset
    y = -obj_x
    return x, y, z


if __name__ == '__main__':
    obj_list = input("输入相机坐标:").split(',')
    obj_x = float(obj_list[0][2:])
    obj_y = float(obj_list[1][2:])
    obj_z = float(obj_list[2][2:])

    z = obj_z - 0.29
    x = obj_y
    y = -obj_x

    print([x - 0.070, y, z])
    print(translate(obj_x, obj_y, obj_z))