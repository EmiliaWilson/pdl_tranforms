import numpy as np
import astropy.units as units
import Transform

data = np.array((1, 2, 3), dtype=np.float64)
m = np.array(([1, 0, 1], [2, 1, 0], [0, 3, 1]))
params = {'matrix': m, 'rot': None, 'scale': None, 'pre': None, 'post': None, 'dims': None}
t_lin = Transform.t_linear(input_coord=['input', 'coord'], input_unit=units.meter, output_coord=['output', 'coord'],
                           output_unit=units.centimeter, parameters=params, reverse_flag=0)
print(f"apply: {t_lin.apply(data)}\nApply with backwards: {t_lin.apply(data, backwards=1)}\n")

params = {'matrix': None, 'rot': np.array((45, 14, 75)), 'scale': None, 'pre': None, 'post': None, 'dims': None}
t_lin2 = Transform.t_linear(input_coord=['input', 'coord'], input_unit=units.meter,
                            output_coord=['output', 'coord'],
                            output_unit=units.centimeter, parameters=params, reverse_flag=0)

print(f"apply: {t_lin2.apply(data)}\nApply with backwards: {t_lin2.apply(data, backwards=1)}\n")
print(t_lin2)
print("\nWith Reverse Flag")
params = {'matrix': None, 'rot': np.array((45, 14, 75)), 'scale': None, 'pre': None, 'post': None, 'dims': None}
t_lin3 = Transform.t_linear(input_coord=['input', 'coord'], input_unit=units.meter,
                            output_coord=['output', 'coord'],
                            output_unit=units.centimeter, parameters=params, reverse_flag=1)

print(f"apply without backwards: {t_lin3.apply(data)}\nApply with backwards: {t_lin3.apply(data, backwards=1)}\n")
print(t_lin3)
print("\nrotation matrix and matrix")
rot_m = np.array(([2, 0, 0], [0, 2, 0], [0, 0, 2]))
params = {'matrix': m, 'rot': rot_m, 'scale': None, 'pre': None, 'post': None, 'dims': None}
t_lin3 = Transform.t_linear(input_coord=['input', 'coord'], input_unit=units.meter,
                            output_coord=['output', 'coord'],
                            output_unit=units.centimeter, parameters=params, reverse_flag=0)

print(f"apply without backwards: {t_lin3.apply(data)}\nApply with backwards: {t_lin3.apply(data, backwards=1)}\n")
print(t_lin3)
