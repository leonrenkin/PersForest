import gudhi as gd

from simplicial import SimplicialFiltration
from _validation import _validate_filtration

def alpha_filtration(point_cloud):
    simplex_tree = gd.AlphaComplex(points=point_cloud).create_simplex_tree(output_squared_values=False)
    filtration_list = simplex_tree.get_filtration()
    return SimplicialFiltration(filtration_list=filtration_list)