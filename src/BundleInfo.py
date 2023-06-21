import itertools
from functools import cached_property

from scipy import stats
import numpy as np
import pandas as pd
import dipy

from evals import binary_mask, bundle_density_map

class BundleInfo:
    '''Bundle shape metrics, as defined in [Yeh, 2020]'''

    def __init__(self, bundle_data, ref_image, voxel_size=2):
        self.voxel_size = voxel_size
        self.data = bundle_data
        self.ref_image = ref_image
        self.n_lines = len(self.data)
        self.n_points = np.array([len(i) for i in self.data])
        
        self.voxel_size_check()
        self.summary_report()
    
    @cached_property
    def streamline_lengths(self):
        '''Returns the lengths for all streamline in the bundle.'''
        return np.array([i for i in dipy.tracking.utils.length(list(self.data))])

    @cached_property
    def bundle_length(self):
        '''Returns length of bundle, calculated as mean of streamline lengths.'''
        return stats.trim_mean(self.streamline_lengths, 0.1)
        # return self.streamline_lengths.mean()
    
    def streamline_span(self, i):
        line = self.data[i]
        return np.linalg.norm(line[0]-line[-1])
    
    @cached_property
    def bundle_span(self):
        '''Returns span of bundle, calculated as mean of streamline span.'''
        spans = [self.streamline_span(i) for i in range(self.n_lines)]
        return stats.trim_mean(spans, 0.1)
        # return np.array(spans).mean()
    
    @cached_property
    def bundle_curl(self):
        '''Returns curl of bundle, calculated as bundle length/span.'''
        return float(self.bundle_length)/self.bundle_span
    
    def voxel_size_check(self):
        '''
            Checks if distance between consecutive points in all streamlines 
            are smaller than voxel size.
        '''
        check = []
        for i in range(self.n_lines):
            pointwise_lens = np.array([np.linalg.norm(self.data[i][j]-self.data[i][j+1]) \
                                  for j in range(self.n_points[i]-1)])
            check.append((pointwise_lens<self.voxel_size).all())
        return np.array(check).all()
    
    @cached_property
    def density_map(self):
        '''Returns the density map of bundle given reference image.'''
        return bundle_density_map(self.data, self.ref_image)
    
    @cached_property
    def density_mask(self):
        '''Returns the density mask of bundle give reference image.'''
        return binary_mask(self.density_map)
    
    @cached_property
    def bundle_volume(self):
        '''Returns volume of bundle, calculated as number of unique 1x1x1 mm3 
            voxels after voxelization'''
        return np.power(self.voxel_size, 3)*np.sum(self.density_map>0)
    
    @cached_property
    def bundle_diameter(self):
        '''Returns diameter of bundle, calculated 2*sqrt(volume/(pi*length))'''
        return 2*float(np.sqrt(self.bundle_volume/(np.pi*self.bundle_length)))
    
    @cached_property
    def bundle_elongation(self):
        '''Returns bundle elongation, calculated as bundle length/diameter'''
        return float(self.bundle_length)/self.bundle_diameter
        
    @staticmethod
    def is_surface_voxel(vox):
        '''Returns if a given voxel is surface voxel'''
        if vox[0] != 0 and 0 in vox[1:]:
            return 1
        return 0

    @cached_property
    def surface_voxels(self):
        '''Returns surface voxel mask.'''
        dim = len(self.density_mask.shape)       # number of dimensions
        offsets = [0, -1, 1]     # offsets, 0 first so the original entry is first 
        columns = []
        for shift in itertools.product(offsets, repeat=dim):   # equivalent to dim nested loops over offsets
            columns.append(np.roll(self.density_mask, shift, np.arange(dim)).ravel())
        neighbors = np.stack(columns, axis=-1)
        return np.apply_along_axis(BundleInfo.is_surface_voxel, 1, neighbors)
    
    @cached_property
    def bundle_surface_area(self):
        '''Returns bundle surface area, calculated from volumized data.'''
        n_surface_voxels = np.count_nonzero(self.surface_voxels)
        return (float(n_surface_voxels)*np.power(self.voxel_size, 2))
    
    @cached_property
    def bundle_irregularity(self):
        '''Returns bundle irregularity, calculated as surface aera / (pi * diameter * length)'''
        return self.bundle_surface_area / (np.pi * self.bundle_diameter * self.bundle_length)
    
    def summary_report(self):
        '''Saves summary report with all shape metrics'''
        round_factor = 4 # number of decimal points to keep
        self.shape_report = {}
        self.shape_report['n_lines'] = self.n_lines
        self.shape_report['n_points'] = len(self.data.get_data())
        self.shape_report['n_avg_points'] = int(stats.trim_mean(self.n_points, 0.1))
        self.shape_report['length'] = np.round(self.bundle_length, round_factor)
        self.shape_report['span'] = np.round(self.bundle_span, round_factor)
        self.shape_report['curl'] = np.round(self.bundle_curl, round_factor)
        self.shape_report['volume'] = self.bundle_volume
        self.shape_report['diameter'] = np.round(self.bundle_diameter, round_factor)
        self.shape_report['elongation'] = np.round(self.bundle_elongation, round_factor)
        self.shape_report['surface_area'] = np.round(self.bundle_surface_area, round_factor)
        self.shape_report['irregularity'] = np.round(self.bundle_irregularity, round_factor)

        self.shape_report = pd.DataFrame([self.shape_report])
        
    def print_summary_report(self):
        '''Prints summary report with a shape metrics.'''
        print(f'---Shape Summary Report---')
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