
from dataclasses import dataclass



#todo: set up Chain class such that it also works for cubical complexes
@dataclass(slots=True)
class Chain:
    signed_simplices: Set[tuple]       # oriented codimension-one simplices, stored as (simplex, orientation)
    active_start: float = float("-inf")
    active_end: float   = float("-inf")
    interior_available: bool = False
    interior: Set[tuple] | None = None
    type: Optional[str] = None 




@dataclass(slots=True)
class SignedChain(Chain): 
    """
    Oriented simplicial chain used as a cycle representative.

    Attributes
    ----------
    signed_simplices : set[tuple]
        Oriented codimension-one simplices, stored as
        ``(simplex_tuple, orientation)``.
    active_start, active_end : float
        Half-open filtration interval ``[active_start, active_end)`` on which
        this representative is active.
    interior_available : bool
        True when ``interior`` has been computed.
    interior : set[tuple] | None
        Oriented full-dimensional simplices whose boundary gives the chain's
        tracked interior.
    """
    signed_simplices: Set[tuple]        # oriented codimension-one simplices, stored as (simplex, orientation)
    active_start: float = float("-inf")
    active_end: float   = float("-inf")
    interior_available: bool = False
    interior: Set[tuple] | None = None        # oriented (d+1)-simplices, stored as (simplex, orientation)
    
    def cancel_simplex(self, simplex: list[int]) -> "SignedChain":
        """
        Remove a simplex that appears with both orientations in this chain.

        Parameters
        ----------
        simplex : list[int]
            Simplex to delete from the chain.

        Returns
        -------
        SignedChain
            Chain with the simplex removed in both orientations.
        """
        return SignedChain(signed_simplices= self.signed_simplices.difference({(tuple(simplex),1),(tuple(simplex),-1)}) )
    
    def unsigned(self) -> "SignedChain":
        """
        Return a new SignedChain where simplices that appear with opposite
        orientations cancel out.

        Returns
        -------
        SignedChain
            Chain without simplices that appear with both orientations.
        """
        # Aggregate orientations per underlying simplex.
        coeffs: Dict[tuple, int] = {}
        for simplex, orientation in self.signed_simplices:
            coeffs[simplex] = coeffs.get(simplex, 0) + int(orientation)

        cleaned: Set[tuple] = set()
        for simplex, c in coeffs.items():
            if c > 0:
                cleaned.add((simplex, 1))
            elif c < 0:
                cleaned.add((simplex, -1))
            # If c == 0, the +1 and -1 copies cancel.
        
        return SignedChain(
            signed_simplices=cleaned,
            active_start=self.active_start,
            active_end=self.active_end,
            interior_available=self.interior_available,
            interior=self.interior
        )

    def only_double_simplices(self) -> "SignedChain":
        """
        Return a new SignedChain where only simplices that appear with opposite
        orientations are preseverd.

        Returns
        -------
        SignedChain
            Chain with only simplices that appear with both orientations.
        """
        # aggregate orientations per underlying simplex
        coeffs: Dict[tuple, int] = {}
        for simplex, orientation in self.signed_simplices:
            coeffs[simplex] = coeffs.get(simplex, 0) + 1

        cleaned: Set[tuple] = set()
        for simplex, c in coeffs.items():
            if c == 2:
                cleaned.add((simplex, 1))
                cleaned.add((simplex, -1))
            elif c == 1:
                continue
            else: 
                raise ValueError("Each simplex should only appear once or twice.")
        
        return SignedChain(
            signed_simplices=cleaned,
            active_start=self.active_start,
            active_end=self.active_end,
        )

    def segments(self, point_cloud: NDArray):
        """
        Convert oriented edges in the chain to geometric segments.

        Parameters
        ----------
        point_cloud : ndarray
            Ambient coordinates for simplex vertices.

        Returns
        -------
        list[np.ndarray]
            List of 2xD arrays representing oriented edge segments.
        """
        segments = []
        for signed_simplex in self.signed_simplices:
            if signed_simplex[1]==1:
                segments.append(np.array(point_cloud[list(signed_simplex[0])]))
            else:
                segments.append(np.array(point_cloud[list(reversed(signed_simplex[0]))]))
        return segments
    
    def simplices(self, signed) -> list[list[int]]:
        if not signed:
            chain = self.unsigned()
        else:
            chain = self
        return [simplex for simplex, orientation in chain.signed_simplices]

    def dim(self):
        """Return the topological dimension of the simplices in this chain."""
        for signed_simplex in self.signed_simplices:
            return len(signed_simplex[0])-1
    
    def vertex_coordinates(self, point_cloud: NDArray, signed = True) -> NDArray[np.float64]:
        """
        Extract the coordinates of the vertices in this SignedChain.

        Parameters
        ----------
        point_cloud : ndarray, shape (n_points, dim)
            Ambient point cloud.
        signed : bool
            If True, use the stored signed simplices. If False, first cancel
            opposite-oriented copies via ``unsigned()``.

        Returns
        -------
        ndarray, shape (n_vertices, dim)
            Coordinates of the vertices in the chain.
        """
        vertex_indices: Set[int] = set()
        if signed: 
            signed_simplices = self.signed_simplices
        else:
            signed_simplices = self.unsigned().signed_simplices

        for signed_simplex in signed_simplices:
            simplex, _ = signed_simplex
            vertex_indices.update(simplex)

        coords = np.array([point_cloud[i] for i in sorted(vertex_indices)], dtype=float)
        return coords
