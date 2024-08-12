import glob
import sys
import os
import numpy as np
import os.path as osp
import mcubes

def get_shape_df(path):
    dims = np.fromfile(path, np.uint64, 3)
    df = np.fromfile(path, np.float32, offset=3 * 8).reshape(dims)
    return df


if __name__ == '__main__':

    base_path = "/mnt/proj74/rhchu/dataset/3d-epn/shapenet_dim32_df"
    cls = '03001627'
    out_path = "/mnt/proj74/rhchu/dataset/3d-epn/vis_df_gt"

    df_path = osp.join(base_path, cls)
    df_files = os.listdir(df_path)
    for df_file in df_files:
        df_name = df_file[:-3]
        df = get_shape_df(osp.join(df_path, df_file))
        out_cls_path = osp.join(out_path, cls)
        os.makedirs(out_cls_path, exist_ok=True)
        tdf = np.clip(df, 0, 3)
        out_file = osp.join(out_cls_path, df_name + '.obj')
        vertices, traingles = mcubes.marching_cubes(tdf, 0.5)
        mcubes.export_obj(vertices, traingles, out_file)
        print(f"Save {out_file}!")




