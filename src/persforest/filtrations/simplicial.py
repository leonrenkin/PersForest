def key(simplex):
    """
    Return a canonical, orientation-free key for a simplex.

    Parameters
    ----------
    simplex : iterable[int]
        Vertex ids of the simplex.

    Returns
    -------
    tuple
        Sorted tuple of vertex ids, suitable for hashing regardless of
        orientation.
    """
    return tuple(sorted(simplex))

class SimplicialFiltration:

    def __init__(self, filtration_list):
        self.filtration_list = filtration_list
        return

    def remove_low_dim_simplices(self,dim: int):
        "only keeps simplices up to dimension dim-1"
        self.filtration_list = [(simplex,filt_val) for simplex, filt_val in self.filtration_list if len(simplex) >= dim]
        return

    def signed_boundary(self, simplex):
        """
        Compute the oriented boundary of a simplex.

        Parameters
        ----------
        simplex : list[int]
            Vertices of the simplex whose boundary is computed.
        orientation : int
            Orientation (+1 or -1) of the simplex.

        Returns
        -------
        set[tuple]
            Set of oriented faces stored as (simplex_tuple, orientation).
        """
        if len(simplex)<2:
            raise ValueError(f"Error for simplex {simplex}: Boundaries for 1-simplices should not occur")

        return tuple( (simplex[:i] + simplex[i + 1:], (-1) ** i) for i in range(len(simplex)))

    def faces(self, simplex):
        return list(itertools.combinations(simplex, len(simplex)-1 ))

    def dim(self,simplex):
        return len(simplex)-1

    def simplex_oriention(self,simplex, point_cloud):
        from geometry.simplicial import simplex_orientation
        return simplex_orientation(simplex=simplex, point_cloud=point_cloud)
    