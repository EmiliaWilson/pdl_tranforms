import numpy as np
import pandas as pd
import astropy.units as units
from abc import ABC, abstractmethod
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interpn
from scipy.interpolate import RegularGridInterpolator
import copy


def ndcoords(*dims):
    grid_size = []
    if type(dims[0]) is tuple or type(dims[0]) is list or type(dims[0]) is np.ndarray:
        for i in range(len(dims[0])):
            grid_size.append(range(dims[0][i]))
    else:
        for i in range(len(dims)):
            print(type(dims[i]), dims[i])
            grid_size.append(range(dims[i]))

    out = np.mgrid[grid_size].transpose()
    # arr = arr.astype('float64')
    out = out.astype('float64')
    return out


class Transform(ABC):

    def __init__(self, name, input_coord, input_unit,
                 output_coord, output_unit, parameters,
                 reverse_flag, input_dim=None,
                 output_dim=None):
        """
        :type name: str
        :type input_coord: list
        :type input_unit: astropy.units
        :type output_coord: np.array
        :type output_unit: astropy.units
        :type parameters: dict

        :type reverse_flag: bool
        :type input_dim: int
        :type output_dim: int
        """
        self.name = name
        self.input_coord = input_coord
        self.input_unit = input_unit
        self.output_coord = output_coord
        self.output_unit = output_unit
        self.parameters = parameters
        self._non_invertible = 0
        self.reverse_flag = reverse_flag
        self.input_dim = input_dim
        self.output_dim = output_dim

    def inverse(self):
        pass

    @abstractmethod
    def apply(self, data, backward=0):
        pass

    def invert(self, data):
        return self.apply(data, backward=1)

    def map(self, data=None, template=None, pdl=None, opts=None):
        # if template is not None:
        #     self.output_dim = template
        # else:
        self.output_dim = data.shape
        out = np.empty(shape=self.output_dim, dtype=np.float64)
        dd = out.shape
        ndc = ndcoords(dd)
        ndc = ndc.reshape((ndc.shape[0] * ndc.shape[1], ndc.shape[2]))
        idx = self.apply(ndc, backward=1)
        pixel_grid = [np.arange(x) for x in data.shape]
        x = interpn(points=pixel_grid, values=data, method='linear', xi=idx.flatten(), bounds_error=False,
                    fill_value=0)
        out[:] = x.reshape(self.output_dim)
        return out

    def match(self, pdl, opts=None):
        return self.map(pdl=pdl, opts=opts)

    def __parse(self, defaults, uopts=None):
        return_dict = defaults.copy()
        if uopts is None:
            return return_dict
        for k in defaults.keys():
            for r in uopts.keys():
                if k == r and uopts[r] is not None:
                    return_dict[k] = uopts[r]

        return return_dict


# f(g(h(x))) == composition([h, g, f]) or composition([f, g, h])
# look at function composition in python to find standard but it seems like second is more common.
class t_compose(Transform):
    def __init__(self, t_list, input_coord=None, input_unit=None, output_coord=None, output_unit=None,
                 parameters=None, reverse_flag=0, input_dim=None, output_dim=None):
        self.func_list = []
        compose_name = ""
        for tform in t_list:
            if type(tform) is t_compose:
                self.func_list.extend(tform.func_list)
                compose_name += tform.name
            else:
                self.func_list.append(tform)
                if tform.reverse_flag == 1:
                    compose_name += f" x {tform.name} inverse"
                else:
                    compose_name += f" x {tform.name}"
        super().__init__(name=compose_name, input_coord=input_coord, input_unit=input_unit, output_coord=output_coord,
                         output_unit=output_unit, parameters=parameters, reverse_flag=reverse_flag, input_dim=input_dim,
                         output_dim=output_dim)

    def apply(self, data, backward=0):
        out_data = copy.deepcopy(data)
        if backward:
            for tform in reversed(self.func_list):
                out_data = tform.apply(out_data, backward=1)

        else:
            for tform in reversed(self.func_list):
                out_data = tform.apply(out_data)

        return out_data


class t_identity(Transform):
    """
    Return Copy of OG data with apply
    """

    def __init__(self, name="Identity", input_coord=None, input_unit=None, output_coord=None,
                 output_unit=None, parameters=None,
                 reverse_flag=None, input_dim=0, output_dim=0):
        super().__init__(name=name, input_coord=input_coord, input_unit=input_unit, output_coord=output_coord,
                         output_unit=output_unit, parameters=parameters,
                         reverse_flag=reverse_flag, input_dim=input_dim, output_dim=output_dim)

    def apply(self, data, backward=0):
        return copy.deepcopy(data)


class t_linear(Transform):
    """
    parameter is a dict with the keys:

    matrix: a numpy.array. The transformation matrix. It does not even have to be square, if you want to change
        the dimensionality of your input. If it is invertible (note: must be square for that), then you automagically
        get an inverse transform too.

    rot: a rotation angle in degrees, another numpy.array. If it is one value, it is a scalar.

    scale: A scaling vector, or a scalar. another numpy.array or a scalar

    pre: The vector to be added to the data before they get multiplied by the matrix
        (equivalent of CRVAL in FITS, if you are converting from scientific to pixel units).

    post: The vector to be added to the data after it gets multiplied by the matrix (equivalent of CRPIX-1 in FITS,
        if youre converting from scientific to pixel units).

    dims:Most of the time it is obvious how many dimensions you want to deal with: if you supply a matrix, it defines
        the transformation; if you input offset vectors in the pre and post options, those define the number of
        dimensions. But if you only supply scalars, there is no way to tell and the default number of dimensions is 2.
        This provides a way to do, e.g., 3-D scaling: just set {s=<scale-factor>, dims=>3}> and you are on your way.

    """

    def __init__(self, input_coord, input_unit,
                 output_coord, output_unit, parameters, reverse_flag,
                 name='t_linear', input_dim=None,
                 output_dim=None):
        # this basic implementation doesn't deal with all the cases you see in PDL. They will be implemented later
        # params = {"matrix": None, "scale": None, "rot": 0, "pre": None, "post": None, "dims": None}

        super().__init__(name, input_coord, input_unit, output_coord,
                         output_unit, parameters,
                         reverse_flag, input_dim, output_dim)

        # Figuring out the number of dimensions to transform, and, if necessary, generate a new matrix
        if self.parameters['matrix'] is not None:
            self.input_dim = self.parameters['matrix'].shape[0]
            self.output_dim = self.parameters['matrix'].shape[1]
        else:
            if self.parameters['rot'] is not None and type(self.parameters['rot']) is np.ndarray:
                if self.parameters['rot'].size == 1:
                    self.input_dim = 2
                    self.output_dim = 2
                elif self.parameters['rot'].size == 3:
                    self.input_dim = 3
                    self.output_dim = 3

            elif self.parameters['scale'] is not None and type(self.parameters['scale']) is np.ndarray:
                self.input_dim = self.parameters['scale'].shape[0]
                self.output_dim = self.parameters['scale'].shape[0]
                # look at craig's response to email about this
            elif self.parameters['pre'] is not None and type(self.parameters['pre']) is np.ndarray:
                self.input_dim = self.parameters['pre'].shape[0]
                self.output_dim = self.parameters['pre'].shape[0]
            elif self.parameters['post'] is not None and type(self.parameters['post']) is np.ndarray:
                self.input_dim = self.parameters['post'].shape[0]
                self.output_dim = self.parameters['post'].shape[0]
            elif self.parameters['dims'] is not None:
                self.input_dim = self.parameters['dims']
                self.output_dim = self.parameters['dims']
            else:
                print("Assuming 2-D transform(set dims options)")
                self.input_dim = 2
                self.output_dim = 2

            self.parameters['matrix'] = np.zeros((self.input_dim, self.output_dim))
            np.fill_diagonal(self.parameters['matrix'], 1)

        # Handle rotation option
        rot = self.parameters['rot']
        if rot is not None:
            if type(rot) is np.ndarray:
                if np.ndim(rot) == 2:
                    # rotation matrix, need to use compose
                    self.parameters['matrix'] = np.matmul(self.parameters['matrix'], rot)
                elif np.size(rot) == 3:
                    rotation = R.from_euler('xyz', [rot[0], rot[1], rot[2]], degrees=True)
                    rot_matrix = rotation.as_dcm()
                    self.parameters['matrix'] = np.matmul(self.parameters['matrix'], rot_matrix)
                else:
                    raise ValueError("Transform.linear got a strange rot option -- giving up.")

            elif rot != 0 and self.parameters['matrix'].shape[0] > 1:
                theta = np.deg2rad(rot)
                c, s = np.cos(theta), np.sin(theta)
                if c < 1e-10:
                    c = 0
                if s < 1e-10:
                    s = 0
                rot_matrix = np.array(((c, -s), (s, c)))
                self.parameters['matrix'] = np.matmul(self.parameters['matrix'], rot_matrix)

        # applying scaling. No matrix. Documentation
        if self.parameters['scale'] is not None and type((self.parameters['scale']) is not np.ndarray):

            for j in range(self.parameters['matrix'].shape[0]):
                self.parameters['matrix'][j][j] *= self.parameters['scale']

        elif type(self.parameters['scale']) is np.ndarray:
            if self.parameters['scale'].ndims > 1:
                raise ValueError("Scale only accepts scalars and 1D arrays")
            else:
                # this might be wrong
                for j in range(self.parameters['matrix'].shape[0]):
                    self.parameters['matrix'][j][j] *= self.parameters['scale'][j]

        # need to check for inverse and set inverted flag. Throw error in apply
        self._non_invertible = 0
        try:
            self.inv = np.linalg.inv(self.parameters['matrix'])
        except np.linalg.LinAlgError:
            self.inv = None
            self._non_invertible = 1

    def apply(self, data, backward=0):
        # print(f"data size: {data[0].size}")'
        og_shape = data.shape
        if data.ndim == 1:
            data = data.reshape((1, 1, data.size))
        else:
            data = data.reshape((data.shape[0], 1, data[0].size))
        # print(data.shape)
        if (not backward and not self.reverse_flag) or (backward and self.reverse_flag):
            d = self.parameters['matrix'].shape[0]
            if d > np.shape(data)[2]:
                raise ValueError(f"Linear transform: transform is {np.shape(data)[2]} data only ")

            if self.parameters['pre'] is not None:
                x = copy.deepcopy(data[..., 0:d]) + self.parameters['pre']
            else:
                x = copy.deepcopy(data[..., 0:d])

            out = copy.deepcopy(data)
            # print(f"out shape: {out.shape}")
            if self.parameters['post'] is not None:
                out[..., 0:d] = np.matmul(x, self.parameters['matrix']) + self.parameters['post']
            else:
                out[..., 0:d] = np.matmul(x, self.parameters['matrix'])
                # temp = np.matmul(x[np.newaxis, ...], self.parameters['matrix'])
                # print(np.squeeze(temp, axis=0).shape)
                # out[0:d] = np.squeeze(temp, axis=0)
                # temp = []
                # print(x.shape)
                # for x_len in range(x.shape[0]):
                #     # print(x[x_len])
                #     temp.append(np.matmul(x[x_len], self.parameters['matrix']))
                #     print(f"temp: {temp}")

                # print(f"temp: {temp}")

            out = out.reshape(og_shape)
            return out

        elif not self._non_invertible:

            d = self.inv.shape[0]
            if d > np.shape(data)[2]:
                raise ValueError(f"Linear transform: transform is {np.shape(data)[2]} data only ")

            if self.parameters['pre'] is not None:
                x = copy.deepcopy(data[..., 0:d]) + self.parameters['pre']
            else:
                x = copy.deepcopy(data[..., 0:d])

            out = copy.deepcopy(data)
            if self.parameters['post'] is not None:
                out[..., 0:d] = np.matmul(x, self.inv) + self.parameters['post']
            else:
                out[..., 0:d] = np.matmul(x, self.inv)

            out = out.reshape(og_shape)
            return out
        else:
            print("trying to invert a non-invertible matrix.")

    def __str__(self):
        to_string = f"Transform name: {self.name}\nInput Coord: {self.input_coord}\nInput Unit:{self.input_unit}\n" \
                    f"Output Coord: {self.output_coord}\nOutput Units: {self.output_unit}\n" \
                    f"matrix:{self.parameters['matrix']}\nscale:{self.parameters['scale']}\nrot: " \
                    f"{self.parameters['rot']}\npre: {self.parameters['pre']}\npost: {self.parameters['post']}\n" \
                    f"dims: {self.parameters['dims']}\nReverse Flag: {self.reverse_flag}\n" \
                    f"Non-Invertible: {self._non_invertible}\nInput Dim: {self.input_dim}\nOutput Dim: {self.output_dim}"

        return to_string


class t_radial(Transform):
    """
    parameter is a dict with keys:

    direct: Generate(theta, r) coordinates out (this is the default); incompatible with Conformal. Theta is in Radians,
        and the radial coordinate is in the units of distance in the input plane

    r0: If defined, this floating-point value causes t_radial to generate
        (theta, ln(r/r0)) coordinates out.  Theta is in radians, and the
        radial coordinate varies by 1 for each e-folding of the r0-scaled
        distance from the input origin.  The logarithmic scaling is useful for
        viewing both large and small things at the same time, and for keeping
        shapes of small things preserved in the image.

    origin: Defaults to (0, 0, 0). This is the origin of the expansion, pass in a numpy array.

    u: units, Default is 'radians', this is the angular unit to be used for the azimuth. degrees is the other option
    """
    _RAD2DEG = 180 / np.pi

    def __init__(self, input_coord, input_unit,
                 output_coord, output_unit, parameters, reverse_flag,
                 name='t_radial', input_dim=None,
                 output_dim=None):
        super().__init__(name, input_coord, input_unit, output_coord,
                         output_unit, parameters,
                         reverse_flag, input_dim, output_dim)

        if self.parameters['u'] == 'degrees':
            self._argunit = self._RAD2DEG
        else:
            self._argunit = 1.0

        self.input_dim = 2
        self.output_dim = 2

        if self.parameters['r0']:
            # line 3458
            self.output_coord = ["Azimuth", "Ln radius"]
            self.output_unit = self.parameters['u']
        else:
            self.output_coord = ["Azimuth", "Radius"]
            self.output_unit = self.parameters['u']

    def apply(self, data, backward=0):
        if (not backward and not self.reverse_flag) or (backward and self.reverse_flag):
            out = copy.deepcopy(data)
            d = copy.deepcopy(data)

            d[0:2] -= self.parameters['origin'][0:2]
            d0 = d[0]
            d1 = d[1]

            out[0] = (np.arctan2(-d1, d0) % (2 * np.pi)) * self._argunit
            if self.parameters['r0'] is not None:
                out[1] = 0.5 * np.log((d1 * d1 + d0 * d0) / (self.parameters['r0'] * self.parameters['r0']))
            else:
                out[1] = np.sqrt(d1 * d1 + d0 * d0)

            return out
        else:
            d0 = copy.deepcopy(data[0])
            d1 = copy.deepcopy(self.__dummy(data[1], [0, 2]))
            out = copy.deepcopy(data)

            d0 /= self._argunit
            out[0:2] = np.column_stack((self.__dummy(np.cos(d0), [0, 1]), self.__dummy(-np.sin(d0), [0, 1])))
            if self.parameters['r0'] is not None:
                out[0:2] *= self.parameters['r0'] * np.exp(d1[0])
            else:
                out[0:2] *= d1[0]
            out[0:2] += self.parameters['origin'][0:2]

            return out

    def __dummy(self, data, shape):
        if type(data) is np.ndarray:
            return np.ones((shape[0] + 1, shape[1]), dtype=np.float64) * data[:, np.newaxis]
        else:
            return np.ones((shape[0] + 1, shape[1]), dtype=np.float64) * data
