import numpy as np
from pyglm import glm

vec1 = glm.vec1
vec2 = glm.vec2
vec3 = glm.vec3
vec4 = glm.vec4

cross = glm.cross


def vee(S):
    """
    Vee operator for SO(3): extracts the vector from a skew-symmetric matrix.
    Used in geometric attitude control (Lee 2010).

    For a skew-symmetric matrix S, returns [S[2,1], S[0,2], S[1,0]].
    """
    return np.array([S[2, 1], S[0, 2], S[1, 0]])
