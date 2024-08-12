import glob
import sys
import os
import numpy as np
import os.path as osp

def get_shape_df(path):
    dims = np.fromfile(path, np.uint64, 3)
    df = np.fromfile(path, np.float32, offset=3 * 8).reshape(dims)
    return df

def get_shape_sdf(path):
    dims = np.fromfile(path, np.uint64, 3)
    sdf = np.fromfile(path, dtype = np.float32, offset=3*8).reshape(dims)
    return sdf

if __name__ == '__main__':

    base_df_path = "/mnt/proj74/rhchu/dataset/3d_epn/shapenet_dim32_df"
    out_df_path = "/mnt/proj74/rhchu/dataset/3d_epn/shapenet_dim32_df_npy"
    base_sdf_path = "/mnt/proj74/rhchu/dataset/3d_epn/shapenet_dim32_sdf"
    out_sdf_path = "/mnt/proj74/rhchu/dataset/3d_epn/shapenet_dim32_sdf_npy"
    
    clss = ['02933112', '04530566', '03636649', '02691156', '02958343', '04379243', '04256520', '03001627']

    # Process df files
    for cls in clss:
        df_path = osp.join(base_df_path, cls)
        df_files = os.listdir(df_path)
        for df_file in df_files:
            df_name = df_file[:-3]
            df = get_shape_df(osp.join(df_path, df_file))
            out_cls_path = osp.join(out_df_path, cls)
            os.makedirs(out_cls_path, exist_ok=True)
            out_file = osp.join(out_cls_path, df_name + '.npy')
            np.save(out_file, df)

    # Process sdf files
    for cls in clss:
        sdf_path = osp.join(base_sdf_path, cls)
        sdf_files = os.listdir(sdf_path)
        for sdf_file in sdf_files:
            sdf_name = sdf_file[:-4]
            sdf = get_shape_sdf(osp.join(sdf_path, sdf_file))
            sdf = np.expand_dims(sdf, 0)
            sdf = np.concatenate([np.fabs(sdf), np.sign(sdf)], axis=0)
            out_cls_path = osp.join(out_sdf_path, cls)
            os.makedirs(out_cls_path, exist_ok=True)
            out_file = osp.join(out_cls_path, sdf_name + '.npy')
            np.save(out_file, sdf)



