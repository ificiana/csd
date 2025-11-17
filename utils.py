"""
Mathematical utilities for rigid body dynamics and geometric control.

Theory:
    The vee operator (v: so(3) -> R^3) is the inverse of the hat operator,
    mapping skew-symmetric matrices back to their vector representation.

    For a skew-symmetric matrix S in so(3) (Lie algebra of SO(3)):
        S = [  0  -c   b ]
            [  c   0  -a ]
            [ -b   a   0 ]

    The vee operator extracts the vector:
        vee(S) = [a, b, c]^T

    This is used in geometric attitude control to extract rotation error
    from the matrix expression:
        e_R = vee(R_d^T R - R^T R_d) / 2

    where R is current orientation and R_d is desired orientation.
"""

import numpy as np


def vee(skew_symmetric_matrix):
    """
    Extracts vector from skew-symmetric matrix (vee operator).

    Args:
        skew_symmetric_matrix: 3x3 skew-symmetric matrix in so(3).

    Returns:
        3-element vector [S[2,1], S[0,2], S[1,0]].
    """
    return np.array(
        [
            skew_symmetric_matrix[2, 1],
            skew_symmetric_matrix[0, 2],
            skew_symmetric_matrix[1, 0],
        ]
    )
