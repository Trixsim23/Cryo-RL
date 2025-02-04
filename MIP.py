import numpy as np
import os
from scipy.ndimage import affine_transform
import scipy.io
import matplotlib.pyplot as plt
from scipy.interpolate import interpn

class MaximumIntensityProjection:
    def __init__(self, source_to_detector_distance, drr_size, drr_voxel_dims):
        """
        Initializes the MaximumIntensityProjection object.

        Parameters:
        source_to_detector_distance: float
            Distance from the x-ray source to the detector (in mm).
        drr_size: tuple of 2 ints
            The size of the generated DRR image (width, height).
        drr_voxel_dims: tuple of 2 floats
            The dimensions of each voxel in the DRR (in mm).
        """
        self.source_to_detector_distance = source_to_detector_distance
        self.drr_size = drr_size
        self.drr_voxdims = drr_voxel_dims

    def project(self, vol, position):
        """
        Generates a digitally reconstructed radiograph (DRR) using maximum intensity projection.

        Parameters:
        vol: 3D numpy array
            The volumetric image to project.
        position: tuple of 3 floats
            The position of the image volume in the coordinate system. The position is defined as:
            (px, py, pz), where px, py, and pz are the displacements of the center of the volume
            in the x, y, and z directions (in mm).

        Returns:
        A 2D numpy array representing the DRR image.
        """

        # centering the volume coordinates
        vol_i = np.linspace(-vol.shape[0]/2+0.5,
                            vol.shape[0]/2-0.5, vol.shape[0])+position[0]
        vol_j = self.source_to_detector_distance - \
            np.linspace(vol.shape[1]-0.5, 0.5, vol.shape[1])+position[1]
        vol_k = np.linspace(-vol.shape[2]/2+0.5,
                            vol.shape[2]/2-0.5, vol.shape[2])+position[2]

        # detector:
        [drr_i, drr_j] = np.meshgrid(
            np.linspace(-self.drr_size[0]/2+0.5, self.drr_size[0] /
                        2-0.5, self.drr_size[0])*self.drr_voxdims[0],
            np.linspace(-self.drr_size[1]/2+0.5, self.drr_size[1] /
                        2-0.5, self.drr_size[1])*self.drr_voxdims[1],
            indexing='ij')

        # get the range of r
        vol_ds = np.sqrt(
            sum([x**2 for x in np.meshgrid(vol_i, vol_j, vol_k, indexing='ij')]))
        drr_ds = np.sqrt(drr_i**2+drr_j**2+self.source_to_detector_distance**2)
        r_max = max([vol_ds.max(), drr_ds.max()])
        r_min = min([vol_ds.min(), drr_ds.min()])
        n_samples = int(np.ceil(1.5*(r_max-r_min)))
        # get spehrical coordinates
        az = np.arctan2(drr_i, drr_j)[..., np.newaxis]
        el = np.arctan2(self.source_to_detector_distance,
                        np.sqrt(drr_i**2 + drr_j**2))[..., np.newaxis]
        r = np.reshape(np.linspace(r_min, r_max, n_samples), (1, 1, n_samples))
        # convert back to cartesian
        sample_z = r * np.cos(el) * np.cos(az)
        sample_y = r * np.cos(el) * np.sin(az)
        sample_x = r * np.sin(el)

        # interpolation get sample values
        samples = interpn(
            (vol_i, vol_j, vol_k),
            vol,
            np.stack([sample_y, sample_x, sample_z], axis=3),
            method='linear', bounds_error=False, fill_value=0.0,
        )

        # compute DDR
        DRR = np.amax(samples, axis=2)

        return DRR
