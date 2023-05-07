import numpy as np
import pandas as pd
import itertools
import dipy

class BundleShape:
    def __init__(self, bundle_data, name="bundle"):
        self.voxel_size = 1
        self.data = bundle_data
        self.name = name
        self.n_lines = len(self.data)
        self.n_points = np.array([len(i) for i in self.data])
        
        self.to_voxel()
        self.to_volume()
        self.count_surface_voxels()
        self.summary_report()
        
    def _streamline_lengths(self):
        return np.array([i for i in dipy.tracking.utils.length(list(self.data))])
    
    def streamline_length(self, i):
        return self._streamline_lengths()[i]
    
    def bundle_length(self):
        '''Return length of bundle, calculated as mean of streamline lengths'''
        return self._streamline_lengths().mean()
    
    def streamline_span(self, i):
        line = self.data[i]
        return np.linalg.norm(line[0]-line[-1])
    
    def bundle_span(self):
        '''Return span of bundle, calculated as mean of streamline span'''
        spans = [self.streamline_span(i) for i in range(self.n_lines)]
        return np.array(spans).mean()
    
    def bundle_curl(self):
        '''Return span of bundle, calculated as bundle length/span'''
        return float(self.bundle_length())/self.bundle_span()
    
    def voxel_size_check(self):
        '''Check if distance between consecutive points in all streamlines 
            are smaller than voxel size'''
        check = []
        for i in range(self.n_lines):
            pointwise_lens = np.array([np.linalg.norm(self.data[i][j]-self.data[i][j+1]) \
                                  for j in range(self.n_points[i]-1)])
            check.append((pointwise_lens<self.voxel_size).all())
        return np.array(check).all()
    
    def to_voxel(self):
#         assert self.voxel_size_check()==True, "Need to resample streamlines."
        self.scale_factor = 2
        self.voxelized = [np.unique(np.rint(self.data[i]*self.scale_factor), axis=0) \
                     for i in range(self.n_lines)]
        self.points = np.unique(np.concatenate(self.voxelized), axis=0).astype(int)
    
    def bundle_volume(self):
        '''Return volume of bundle, calculated as number of unique 1x1x1 mm3 
            voxels after voxelization'''
        return len(self.points)/(np.power(self.scale_factor, 3))
    
    def bundle_diameter(self):
        '''Return diameter of bundle, calculated 2*sqrt(volume/(pi*length))'''
        return 2*float(np.sqrt(self.bundle_volume()/(np.pi*self.bundle_length())))
    
    def bundle_elongation(self):
        '''Return bundle elongation, calculated as bundle length/diameter'''
        return float(self.bundle_length())/self.bundle_diameter()
    
    def to_volume(self):
        coord_min = self.points.min(axis=0)
        coord_max = self.points.max(axis=0)
        volume = np.zeros((coord_max[0]-coord_min[0]+1, 
                           coord_max[1]-coord_min[1]+1,
                           coord_max[2]-coord_min[2]+1))
        idx = self.points-coord_min
        volume[idx[:,0],idx[:,1],idx[:,2]]=1
        assert np.count_nonzero(volume) == len(self.points), "Wrong volume calculation!"
        self.volume = volume
        
    @staticmethod
    def is_surface_voxel(vox):
        if vox[0] != 0 and 0 in vox[1:]:
            return 1
        return 0

    def count_surface_voxels(self):
        dim = len(self.volume.shape)       # number of dimensions
        offsets = [0, -1, 1]     # offsets, 0 first so the original entry is first 
        columns = []
        for shift in itertools.product(offsets, repeat=dim):   # equivalent to dim nested loops over offsets
            columns.append(np.roll(self.volume, shift, np.arange(dim)).ravel())
        neighbors = np.stack(columns, axis=-1)
        self.n_surface_voxels = np.count_nonzero(np.apply_along_axis(BundleShape.is_surface_voxel, 1, neighbors))
    
    def bundle_surface_area(self):
        '''Return bundle surface area, calculated from volumized data'''
        return (float(self.n_surface_voxels)*np.power(self.voxel_size, 2))/(np.power(self.scale_factor, 2))
    
    def bundle_irregularity(self):
        '''Return bundle irregularity, calculated as surface aera / (pi * diameter * length)'''
        return self.bundle_surface_area() / (np.pi * self.bundle_diameter() * self.bundle_length())

#     @staticmethod
#     def get_3d_neighbors_idx():
#         m = [-1, 0, 1]
#         neighbor_idx = np.stack(np.meshgrid(m, m, m), axis=-1).reshape(-1, 3)
#         return np.delete(neighbor_idx, 13, axis=0)

#     @staticmethod
#     def is_surface(arr, x, y, z):
#         nx, ny, nz = arr.shape
#         nb_idx = []
#         nb_idx_rel = BundleShape.get_3d_neighbors_idx()
#         for idx in nb_idx_rel:
#             if 0 <= idx[0]+x < nx and 0 <= idx[1]+y < ny and 0 <= idx[2]+z < nz:
#                 nb_idx.append((idx[0]+x, idx[1]+y, idx[2]+z))
#         nb_vals = [arr[i] for i in nb_idx]
#         if arr[x,y,z] != 0 and 0 in nb_vals:
#             return True
#         return False
    
#     def get_surface_voxels(self):
#         surface_volume = self.volume.copy()
#         x, y, z = self.volume.shape
#         for i in range(x):
#             for j in range(y):
#                 for k in range(z):
#                     surface_volume[i,j,k] = self.is_surface(self.volume, i, j, k)
#         print(surface_volume.shape)
#         return surface_volume, np.count_nonzero(surface_volume)
    
    def summary_report(self):
        round_factor = 4 # number of decimal points to keep
        self.shape_report = {}
        self.shape_report['name'] = self.name
        self.shape_report['n_lines'] = self.n_lines
        self.shape_report['n_avg_points'] = int(self.n_points.mean())
        self.shape_report['length'] = np.round(self.bundle_length(), round_factor)
        self.shape_report['span'] = np.round(self.bundle_span(), round_factor)
        self.shape_report['curl'] = np.round(self.bundle_curl(), round_factor)
        self.shape_report['volume'] = self.bundle_volume()
        self.shape_report['diameter'] = np.round(self.bundle_diameter(), round_factor)
        self.shape_report['elongation'] = np.round(self.bundle_elongation(), round_factor)
        self.shape_report['surface_area'] = np.round(self.bundle_surface_area(), round_factor)
        self.shape_report['irregularity'] = np.round(self.bundle_irregularity(), round_factor)

        self.shape_report = pd.DataFrame([self.shape_report])
        
    def print_summary_report(self):
        print(f'---Shape Descriptors for Bundle {self.name}---')
        print(f"Number of lines:\t\t {self.n_lines}")
        print(f"Average number of points:\t {self.shape_report['n_avg_points'].values[0]}")
        print(f"Bundle length:\t\t\t {self.shape_report['length'].values[0]}")
        print(f"Bundle span:\t\t\t {self.shape_report['span'].values[0]}")
        print(f"Bundle curl:\t\t\t {self.shape_report['curl'].values[0]}")
        print(f"Bundle volume:\t\t\t {self.shape_report['volume'].values[0]}")
        print(f"Bundle diameter:\t\t {self.shape_report['diameter'].values[0]}")
        print(f"Bundle elongation:\t\t {self.shape_report['elongation'].values[0]}")
        print(f"Bundle surface area:\t\t {self.shape_report['surface_area'].values[0]}")
        print(f"Bundle irregularity:\t\t {self.shape_report['irregularity'].values[0]}")