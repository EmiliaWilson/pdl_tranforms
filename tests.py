import numpy as np
import astropy.units as units
import Transform
import unittest


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

        data_out = self.t_rad1.apply(self.data, backward=1)
        # print(f"inverted: {data_out}")
        self.assertAlmostEqual(data_out[0], 1.0806046)
        self.assertAlmostEqual(data_out[1], -1.682942)
        self.assertAlmostEqual(data_out[2], 3)
        # self.assertTrue(True)

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


class TestMap(unittest.TestCase):
    def test_linear(self):
        from astropy.io import fits
        import matplotlib.pyplot as plt
        hdul = fits.open('C:\\Users\Jake\Desktop\CU_2020\CU_2020\L2_update.fts.gz')
        image_data = hdul[0].data * 1e10
        params = {'matrix': None, 'rot': 90, 'scale': None, 'pre': None, 'post': None, 'dims': 2}
        t_lin = Transform.t_linear(name='t_lin2', input_coord=['input', 'coord'], input_unit=units.meter,
                                   output_coord=['output', 'coord'],
                                   output_unit=units.centimeter, parameters=params, reverse_flag=0)
        t_lin.map(data=image_data)
        self.assertTrue(True)
    #     output_dim = image_data.shape
    #     out = np.empty(shape=output_dim, dtype=np.float64)
    #     dd = out.shape
    #     print(type(dd), dd)
    #     ndc = self.ndcoords(5, 5)
    #     self.assertTrue(True)
    #
    # def ndcoords(*dims):
    #     if type(dims[0]) is tuple:
    #         test_ndindex = np.ndindex(dims[0])
    #     elif type(dims[0]) is list or type(dims[0]) is np.ndarray:
    #         test_ndindex = np.ndindex(tuple(dims[0]))
    #     else:
    #         test_ndindex = np.ndindex(dims)
    #
    #     end_flag = 1
    #     first_loop = 1
    #     out = None
    #     while end_flag:
    #         try:
    #             if first_loop == 1:
    #                 arr1 = np.asarray(test_ndindex.next())
    #                 arr2 = np.asarray(test_ndindex.next())
    #                 out = np.stack((arr1, arr2))
    #                 first_loop = 0
    #
    #             arr = np.asarray(test_ndindex.next())
    #             out = np.vstack((out, arr))
    #         except:
    #             end_flag = 0
    #
    #     if out is None:
    #         print("out is none")
    #         return None
    #     else:
    #         out = np.fliplr(out)
    #         # print(f"dims: {dims}, type: {type(dims)}")
    #         if type(dims[0]) is tuple:
    #             reshape_dim = list(dims[0])
    #             reshape_dim.append(len(dims[0]))
    #         elif type(dims[0]) is list or type(dims[0]) is np.ndarray:
    #             reshape_dim = list(dims[0])
    #             reshape_dim.append(len(dims[0]))
    #         else:
    #             reshape_dim = list(dims)
    #             reshape_dim.append(len(dims))
    #         out = out.reshape(reshape_dim)
    #         return out


if __name__ == '__main__':
    unittest.main()
