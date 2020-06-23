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


if __name__ == '__main__':
    unittest.main()