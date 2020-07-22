import numpy as np
import astropy.units as units
import Transform
import unittest


def ndcoords(*dims):
    grid_size = []
    if type(dims[0]) is tuple or type(dims[0]) is list or type(dims[0]) is np.ndarray:
        for i in range(len(dims[0])):
            grid_size.append(range(dims[0][i]))
    else:
        for i in range(len(dims)):
            print(type(dims[i]), dims[i])
            grid_size.append(range(dims[i]))

    out = np.mgrid[grid_size]

    out = out.astype('float64').transpose()
    return out


def dummy(data, dim):
    data_shape = list(data.shape)
    if dim[0] < -(len(data_shape)+1):
        raise ValueError("For safety, pos < -(data.ndims+1) forbidden in dummy")
    elif dim[0] < 0:
        if dim[0] == -1:
            data_shape.append(dim[1])
        else:
            data_shape.insert(dim[0] + len(data_shape) + 1, dim[1])
    elif dim[0] > len(data_shape) - 1:
        while len(data_shape) - 1 < dim[0] - 1:
            data_shape.append(1)
        data_shape.insert(dim[0], dim[1])
    else:
        data_shape.insert(dim[0], dim[1])
    new_data = np.ones(data_shape, dtype=np.float64)
    if is_broadcastable(new_data.shape, data[..., np.newaxis].shape):
        return new_data * data[..., np.newaxis]
    else:
        return new_data * data


def is_broadcastable(shp1, shp2):
    for a, b in zip(shp1[::-1], shp2[::-1]):
        if a == 1 or b == 1 or a == b:
            pass
        else:
            return False
    return True


class TestLinear(unittest.TestCase):

    def test_matrix(self):
        data = np.array((1, 2, 3), dtype=np.float64)
        m = np.array(([1, 0, 1], [2, 1, 0], [0, 3, 1]))
        params = {'matrix': m, 'rot': None, 'scale': None, 'pre': None, 'post': None, 'dims': None}
        t_lin = Transform.t_linear(name='t_lin', input_coord=['input', 'coord'], input_unit=units.meter,
                                   output_coord=['output', 'coord'], output_unit=units.centimeter, parameters=params,
                                   reverse_flag=0)
        data_out = t_lin.apply(data)
        # self.assertAlmostEqual(data_out[0], np.array((5, 11, 4)))
        self.assertAlmostEqual(data_out[0], 5)
        self.assertAlmostEqual(data_out[1], 11)
        self.assertAlmostEqual(data_out[2], 4)
        data_out = t_lin.apply(data, backward=1)
        # self.assertAlmostEqual(data_out, np.array((2.1428571, -0.57142857, 0.85714286)))
        self.assertAlmostEqual(data_out[0], 2.1428571)
        self.assertAlmostEqual(data_out[1], -0.57142857)
        self.assertAlmostEqual(data_out[2], 0.85714286)

    def test_rot(self):
        data = np.array((1, 2, 3), dtype=np.float64)
        params = {'matrix': None, 'rot': np.array((45, 14, 75)), 'scale': None, 'pre': None, 'post': None, 'dims': None}
        t_lin = Transform.t_linear(name='t_lin2', input_coord=['input', 'coord'], input_unit=units.meter,
                                   output_coord=['output', 'coord'],
                                   output_unit=units.centimeter, parameters=params, reverse_flag=0)
        data_out = t_lin.apply(data)
        # self.assertAlmostEqual(data_out, np.array((1.3998327, 2.116057, 2.7500416)), places=5)
        self.assertAlmostEqual(data_out[0], 1.3998327)
        self.assertAlmostEqual(data_out[1], 2.116057, places=4)
        self.assertAlmostEqual(data_out[2], 2.7500416)

        params = {'matrix': None, 'rot': 75, 'scale': None, 'pre': None, 'post': None, 'dims': 2}
        t_lin = Transform.t_linear(name='t_lin2', input_coord=['input', 'coord'], input_unit=units.meter,
                                   output_coord=['output', 'coord'],
                                   output_unit=units.centimeter, parameters=params, reverse_flag=0)
        data_out = t_lin.apply(data)
        # self.assertAlmostEqual(data_out, np.array((2.1906707, -0.44828774, 3)), places=5)
        self.assertAlmostEqual(data_out[0], 2.1906707)
        self.assertAlmostEqual(data_out[1], -0.44828774)
        self.assertAlmostEqual(data_out[2], 3)

        data_2d = np.array(([0, 0], [1, 0], [1, 1], [0, 1]), dtype=np.float64)

        # print(f"data_2d: {data_2d}")
        params = {'matrix': None, 'rot': 90, 'scale': None, 'pre': None, 'post': None, 'dims': 2}
        t_lin_2d = Transform.t_linear(name='t_lin2d', input_coord=['input', 'coord'], input_unit=units.meter,
                                      output_coord=['output', 'coord'],
                                      output_unit=units.centimeter, parameters=params, reverse_flag=0)
        data_out = t_lin_2d.apply(data=data_2d)
        # self.assertAlmostEqual(data_out[0], [0, 0])
        # self.assertAlmostEqual(data_out[1], -0.44828774)
        # self.assertAlmostEqual(data_out[2], 3)
        print(f"out: {data_out}")

    def test_reverse_flag(self):
        data = np.array((1, 2, 3), dtype=np.float64)
        params = {'matrix': None, 'rot': np.array((45, 14, 75)), 'scale': None, 'pre': None, 'post': None, 'dims': 3}
        t_lin = Transform.t_linear(input_coord=['input', 'coord'], input_unit=units.meter,
                                   output_coord=['output', 'coord'],
                                   output_unit=units.centimeter, parameters=params, reverse_flag=1)
        data_out = t_lin.apply(data)
        # self.assertAlmostEqual(data_out, np.array((1.1555176, 1.5803996, 3.1885915)), places=5)
        self.assertAlmostEqual(data_out[0], 1.1555176)
        self.assertAlmostEqual(data_out[1], 1.5803996)
        self.assertAlmostEqual(data_out[2], 3.1885915)

    def test_rot_matrix(self):
        data = np.array((1, 2, 3), dtype=np.float64)
        rot_m = np.array(([2, 0, 0], [0, 2, 0], [0, 0, 2]))
        m = np.array(([1, 0, 1], [2, 1, 0], [0, 3, 1]))
        params = {'matrix': m, 'rot': rot_m, 'scale': None, 'pre': None, 'post': None, 'dims': 3}
        t_lin = Transform.t_linear(name='t_lin4', input_coord=['input', 'coord'], input_unit=units.meter,
                                   output_coord=['output', 'coord'],
                                   output_unit=units.centimeter, parameters=params, reverse_flag=0)
        data_out = t_lin.apply(data)
        # self.assertAlmostEqual(data_out, np.array((10, 22, 8)), places=5)
        self.assertAlmostEqual(data_out[0], 10)
        self.assertAlmostEqual(data_out[1], 22)
        self.assertAlmostEqual(data_out[2], 8)

    def test_all_opts(self):
        data = np.array((1, 2, 3), dtype=np.float64)
        m = np.array(([1, 0, 1], [2, 1, 0], [0, 3, 1]))
        params = {'matrix': m, 'rot': np.array((45, 14, 75)), 'scale': 3, 'pre': 2, 'post': 5, 'dims': 3}
        t_lin = Transform.t_linear(name='t_lin5', input_coord=['input', 'coord'], input_unit=units.meter,
                                   output_coord=['output', 'coord'],
                                   output_unit=units.centimeter, parameters=params, reverse_flag=0)
        data_out = t_lin.apply(data)
        # self.assertAlmostEqual(data_out, np.array((23.689761, 2.6456052, 24.478939)), places=5)
        self.assertAlmostEqual(data_out[0], 23.689761)
        self.assertAlmostEqual(data_out[1], 2.6456052)
        self.assertAlmostEqual(data_out[2], 24.478939, places=5)

    def test_post_pre(self):
        data = np.array(([0, 0], [1, 0], [1, 1], [0, 1]), dtype=np.float64)
        params = {'matrix': None, 'rot': 90, 'scale': None, 'pre': None, 'post': np.array((0, 2000)), 'dims': 2}
        t_lin = Transform.t_linear(name='t_lin5', input_coord=['input', 'coord'], input_unit=units.meter,
                                   output_coord=['output', 'coord'],
                                   output_unit=units.centimeter, parameters=params, reverse_flag=0)
        print(t_lin.apply(data))
        self.assertTrue(True)

    def test_scale(self):
        data = ndcoords(3, 3)
        params = {'matrix': None, 'rot': None, 'scale': np.array([2, 3]), 'pre': None, 'post': None, 'dims': 2}
        t_lin = Transform.t_linear(name='t_lin5', input_coord=['input', 'coord'], input_unit=units.meter,
                                   output_coord=['output', 'coord'],
                                   output_unit=units.centimeter, parameters=params, reverse_flag=0)
        print(t_lin.apply(data))
        self.assertTrue(True)


class TestCompose(unittest.TestCase):
    data = np.array((1, 2, 3), dtype=np.float64)
    m = np.array(([1, 0, 1], [2, 1, 0], [0, 3, 1]))
    params = {'matrix': m, 'rot': None, 'scale': None, 'pre': None, 'post': None, 'dims': None}
    t_lin = Transform.t_linear(name='t_lin', input_coord=['input', 'coord'], input_unit=units.meter,
                               output_coord=['output', 'coord'], output_unit=units.centimeter, parameters=params,
                               reverse_flag=0)

    params = {'matrix': None, 'rot': np.array((45, 14, 75)), 'scale': None, 'pre': None, 'post': None, 'dims': None}
    t_lin2 = Transform.t_linear(name='t_lin2', input_coord=['input', 'coord'], input_unit=units.meter,
                                output_coord=['output', 'coord'],
                                output_unit=units.centimeter, parameters=params, reverse_flag=0)

    rot_m = np.array(([2, 0, 0], [0, 2, 0], [0, 0, 2]))
    params = {'matrix': m, 'rot': rot_m, 'scale': None, 'pre': None, 'post': None, 'dims': 3}
    t_lin4 = Transform.t_linear(name='t_lin4', input_coord=['input', 'coord'], input_unit=units.meter,
                                output_coord=['output', 'coord'],
                                output_unit=units.centimeter, parameters=params, reverse_flag=0)

    params = {'matrix': m, 'rot': np.array((45, 14, 75)), 'scale': 3, 'pre': 2, 'post': 5, 'dims': 3}
    t_lin5 = Transform.t_linear(name='t_lin5', input_coord=['input', 'coord'], input_unit=units.meter,
                                output_coord=['output', 'coord'],
                                output_unit=units.centimeter, parameters=params, reverse_flag=0)

    params = {'matrix': None, 'rot': None, 'scale': None, 'pre': np.array((-1024, -1024)), 'post': None, 'dims': 2}
    t_lin6 = Transform.t_linear(name='t_lin6', input_coord=['input', 'coord'], input_unit=units.meter,
                                output_coord=['output', 'coord'],
                                output_unit=units.centimeter, parameters=params, reverse_flag=0)
    params = {'matrix': None, 'rot': 90, 'scale': None, 'pre': None, 'post': None, 'dims': 2}
    t_lin7 = Transform.t_linear(name='t_lin6', input_coord=['input', 'coord'], input_unit=units.meter,
                                output_coord=['output', 'coord'],
                                output_unit=units.centimeter, parameters=params, reverse_flag=0)

    params = {'matrix': None, 'rot': None, 'scale': None, 'pre': None, 'post': np.array((1024, 1024)), 'dims': 2}
    t_lin8 = Transform.t_linear(name='t_lin6', input_coord=['input', 'coord'], input_unit=units.meter,
                                output_coord=['output', 'coord'],
                                output_unit=units.centimeter, parameters=params, reverse_flag=0)

    params = {'matrix': None, 'rot': 90, 'scale': None, 'pre': np.array((-1024, -1024)), 'post': np.array((1024, 1024)), 'dims': 2}
    t_lin9 = Transform.t_linear(name='t_lin6', input_coord=['input', 'coord'], input_unit=units.meter,
                                output_coord=['output', 'coord'],
                                output_unit=units.centimeter, parameters=params, reverse_flag=0)

    params = {'direct': None, 'r0': None, 'origin': np.array((0, 0)), 'u': 'radians'}
    t_rad1 = Transform.t_radial(name='t_rad1', input_coord=['input', 'coord'], input_unit=units.meter,
                                output_coord=['output', 'coord'], output_unit=units.meter, parameters=params,
                                reverse_flag=0)

    def test_all_transforms(self):
        t_comp = Transform.t_compose([self.t_lin5, self.t_lin2, self.t_lin])
        intermediate_data1 = self.t_lin.apply(self.data)
        intermediate_data2 = self.t_lin2.apply(intermediate_data1)
        final_data = self.t_lin5.apply(intermediate_data2)

        intermediate_data1 = self.t_lin.apply(self.data, backward=1)
        intermediate_data2 = self.t_lin2.apply(intermediate_data1, backward=1)
        final_data_inverted = self.t_lin5.apply(intermediate_data2, backward=1)

        data_out = t_comp.apply(self.data)
        self.assertAlmostEqual(data_out[0], final_data[0])
        self.assertAlmostEqual(data_out[1], final_data[1])
        self.assertAlmostEqual(data_out[2], final_data[2])

        data_out = t_comp.apply(self.data, backward=1)
        self.assertAlmostEqual(data_out[0], final_data_inverted[0])
        self.assertAlmostEqual(data_out[1], final_data_inverted[1])
        self.assertAlmostEqual(data_out[2], final_data_inverted[2])

    def test_with_compose(self):
        t_comp = Transform.t_compose([self.t_lin5, self.t_lin2, self.t_lin])
        compose_with_comp = Transform.t_compose([self.t_lin4, t_comp])
        intermediate_data1 = self.t_lin.apply(self.data)
        intermediate_data2 = self.t_lin2.apply(intermediate_data1)
        intermediate_data3 = self.t_lin5.apply(intermediate_data2)
        final_data = self.t_lin4.apply(intermediate_data3)

        data_out = compose_with_comp.apply(self.data)
        self.assertAlmostEqual(data_out[0], final_data[0])
        self.assertAlmostEqual(data_out[1], final_data[1])
        self.assertAlmostEqual(data_out[2], final_data[2])

    def test_map(self):
        t_comp = Transform.t_compose([self.t_lin8, self.t_lin7, self.t_lin6])
        from astropy.io import fits
        import matplotlib.pyplot as plt

        hdul = fits.open('C:\\Users\Jake\Desktop\CU_2020\CU_2020\L2_update.fts.gz')
        image_data = hdul[0].data * 1e1
        plt.figure()
        plt.imshow(image_data, origin='lower')
        plt.show(block=False)
        self.assertTrue(True)
        map_out = t_comp.map(image_data)
        plt.figure()
        plt.imshow(map_out, origin='lower')
        plt.show(block=False)
        plt.show()


class TestRadial(unittest.TestCase):
    data = np.array((1, 2, 3), dtype=np.float64)
    params = {'direct': None, 'r0': None, 'origin': np.array((0, 0, 0)), 'u': 'radians'}
    t_rad1 = Transform.t_radial(name='t_rad1', input_coord=['input', 'coord'], input_unit=units.meter,
                                output_coord=['output', 'coord'], output_unit=units.meter, parameters=params,
                                reverse_flag=0)
    params = {'direct': None, 'r0': None, 'origin': np.array((0.5, 0.5, 0.5)), 'u': 'radians'}
    t_rad2 = Transform.t_radial(name='t_rad2', input_coord=['input', 'coord'], input_unit=units.meter,
                                output_coord=['output', 'coord'], output_unit=units.meter, parameters=params,
                                reverse_flag=0)
    params = {'direct': None, 'r0': 2.5, 'origin': np.array((0, 0, 0)), 'u': 'radians'}
    t_rad3 = Transform.t_radial(name='t_rad3', input_coord=['input', 'coord'], input_unit=units.meter,
                                output_coord=['output', 'coord'], output_unit=units.meter, parameters=params,
                                reverse_flag=0)
    params = {'direct': None, 'r0': None, 'origin': np.array((0, 0, 0)), 'u': 'degrees'}
    t_rad4 = Transform.t_radial(name='t_rad4', input_coord=['input', 'coord'], input_unit=units.meter,
                                output_coord=['output', 'coord'], output_unit=units.meter, parameters=params,
                                reverse_flag=0)

    def test_basic(self):
        data_out = self.t_rad1.apply(self.data)
        # print(data_out)
        self.assertAlmostEqual(data_out[0], 5.1760366)
        self.assertAlmostEqual(data_out[1], 2.236068)
        self.assertAlmostEqual(data_out[2], 3)

        # data_out = self.t_rad1.apply(self.data, backward=1)
        # # print(f"inverted: {data_out}")
        # self.assertAlmostEqual(data_out[0], 1.0806046)
        # self.assertAlmostEqual(data_out[1], -1.682942)
        # self.assertAlmostEqual(data_out[2], 3)
        # # self.assertTrue(True)

        data_2d = np.array(([0, 0], [1, 0], [1, 1], [0, 1]), dtype=np.float64)
        data_test = np.array((1, 0), dtype=np.float64)
        data_out = self.t_rad1.apply(data_2d, backward=1)
        print(data_out)

    def test_origin(self):
        data_out = self.t_rad2.apply(self.data)
        # print(data_out)
        self.assertAlmostEqual(data_out[0], 5.0341395)
        self.assertAlmostEqual(data_out[1], 1.5811388)
        self.assertAlmostEqual(data_out[2], 3)
        data_out = self.t_rad2.apply(self.data, backward=1)
        # print(f"origin reversed: {data_out}")
        self.assertAlmostEqual(data_out[0], 1.5806046)
        self.assertAlmostEqual(data_out[1], -1.182942)
        self.assertAlmostEqual(data_out[2], 3)

    def test_r0(self):
        data_out = self.t_rad3.apply(self.data)
        # print(f"r0 data: {data_out}")
        self.assertAlmostEqual(data_out[0], 5.1760366)
        self.assertAlmostEqual(data_out[1], -0.11157178)
        self.assertAlmostEqual(data_out[2], 3)
        data_out = self.t_rad3.apply(self.data, backward=1)
        # print(f"r0 invert data: {data_out}")
        self.assertAlmostEqual(data_out[0], 9.9808101)
        self.assertAlmostEqual(data_out[1], -15.5441907, places=5)
        self.assertAlmostEqual(data_out[2], 3)

    def test_degree(self):
        data_out = self.t_rad4.apply(self.data)
        # print(f"degree: {data_out}")
        self.assertAlmostEqual(data_out[0], 296.56505, places=5)
        self.assertAlmostEqual(data_out[1], 2.236068, places=5)
        self.assertAlmostEqual(data_out[2], 3)
        data_out = self.t_rad4.apply(self.data, backward=1)
        # print(f"degree invert: {data_out}")
        self.assertAlmostEqual(data_out[0], 1.9996954)
        self.assertAlmostEqual(data_out[1], -0.034904813)
        self.assertAlmostEqual(data_out[2], 3)

    def test_unit_circle(self):
        data_points = np.array(([0, 0], [1, 0], [np.sqrt(3) / 2, .5], [1 / np.sqrt(2), 1 / np.sqrt(2)]))
        data_out = self.t_rad1.apply(data_points)
        print(data_out)
        self.assertTrue(True)



class TestMap(unittest.TestCase):
    # Autoscaling
    # my rot is counter-clockwise
    def test_linear(self):
        from astropy.io import fits
        import matplotlib.pyplot as plt

        from scipy.interpolate import interpn

        hdul = fits.open('C:\\Users\Jake\Desktop\CU_2020\CU_2020\L2_update.fts.gz')
        image_data = hdul[0].data * 1e10
        plt.figure()
        plt.imshow(image_data, origin='lower')
        plt.show(block=False)
        # print(image_data.shape)

        params = {'matrix': None, 'rot': None, 'scale': np.array([1, 2048/6.28]), 'pre': None, 'post': None,
                  'dims': 2}
        t_lin = Transform.t_linear(name='t_lin', input_coord=['input', 'coord'], input_unit=units.meter,
                                   output_coord=['output', 'coord'],
                                   output_unit=units.centimeter, parameters=params, reverse_flag=0)
        params = {'matrix': None, 'rot': None, 'scale': None, 'pre': np.array((-1024, -1024)),
                  'post': None,
                  'dims': 2}
        t_lin2 = Transform.t_linear(name='t_lin2', input_coord=['input', 'coord'], input_unit=units.meter,
                                   output_coord=['output', 'coord'],
                                   output_unit=units.centimeter, parameters=params, reverse_flag=0)
        params = {'direct': None, 'r0': None, 'origin': np.array((0, 0)), 'u': 'radians'}
        t_rad1 = Transform.t_radial(name='t_rad1', input_coord=['input', 'coord'], input_unit=units.meter,
                                    output_coord=['output', 'coord'], output_unit=units.meter, parameters=params,
                                    reverse_flag=0)
        # print(image_data)
        t_comp = Transform.t_compose([t_lin, t_rad1, t_lin2])
        map_out = t_comp.map(data=image_data)
        # inter1 = t_lin.map(image_data)
        # map_out = t_rad1.map(image_data)
        print(map_out.max())
        print(map_out.min())
        print(map_out.mean())
        plt.figure()
        plt.imshow(map_out.transpose(), origin='lower')
        plt.show(block=False)
        plt.show()

    def test_radial(self):

        params = {'direct': None, 'r0': None, 'origin': np.array((0, 0)), 'u': 'radians'}
        t_rad1 = Transform.t_radial(name='t_rad1', input_coord=['input', 'coord'], input_unit=units.meter,
                                    output_coord=['output', 'coord'], output_unit=units.meter, parameters=params,
                                    reverse_flag=0)

        from astropy.io import fits
        import matplotlib.pyplot as plt

        from scipy.interpolate import interpn

        hdul = fits.open('C:\\Users\Jake\Desktop\CU_2020\CU_2020\L2_update.fts.gz')
        image_data = hdul[0].data
        map_out = t_rad1.map(image_data).transpose()
        print(map_out[1300, 1800])

        self.assertTrue(True)


class TestNDCoords(unittest.TestCase):
    # we need to fix how it returns the transpose instead of the actual coord.
    def test_apply(self):
        ndc = ndcoords(3, 3)
        # print(ndc)
        params = {'matrix': None, 'rot': 30, 'scale': None, 'pre': None, 'post': None, 'dims': 2}
        t_lin = Transform.t_linear(name='t_lin2', input_coord=['input', 'coord'], input_unit=units.meter,
                                   output_coord=['output', 'coord'],
                                   output_unit=units.centimeter, parameters=params, reverse_flag=0)
        params = {'direct': None, 'r0': None, 'origin': np.array((0, 0)), 'u': 'radians'}
        t_rad1 = Transform.t_radial(name='t_rad1', input_coord=['input', 'coord'], input_unit=units.meter,
                                    output_coord=['output', 'coord'], output_unit=units.meter, parameters=params,
                                    reverse_flag=0)
        # print(ndc)
        # ndc = ndc.reshape((np.product(ndc.shape[:-1]), ndc.shape[-1]))
        data = np.array(([0, 0], [1, 0], [1, 1], [0, 1]), dtype=np.float64)
        print(f"apply: {t_rad1.apply(data, backward=1)}")
        # print(f"matmul: {np.matmul(ndc, t_lin.parameters['matrix'])}")

        self.assertTrue(True)

    def test_error(self):
        import copy
        data = ndcoords(3, 3)
        d0 = copy.deepcopy(data[..., 0])
        d1 = copy.deepcopy(dummy(data[..., 1], [2, 2]))
        out = copy.deepcopy(data)
        print(d1)
        test = np.array(([0, 1, 2, 3], [4, 5, 6, 7]))

        # print(dummy(test, np.array((1, 3))))
        # print(dummy(test, np.array((1, 3))).shape)

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
