def are_dict_keys_sorted(d):
    """
    Return True if dict keys are in ascending order (linear time).

    Parameters
    ----------
    d : dict
        Dictionary whose key insertion order is inspected.

    Returns
    -------
    bool
        True if keys appear in ascending order, False otherwise.
    """
    it = iter(d)  # iterates over keys in insertion order
    try:
        prev = next(it)
    except StopIteration:
        return True  # empty dict is trivially sorted

    for k in it:
        if k < prev:
            return False
        prev = k
    return True