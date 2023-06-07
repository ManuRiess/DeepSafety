from scipy.spatial.transform import Rotation
import numpy as np

# Camera data:
# Center point:
cx = None
cy = None
# Focal lengths  F(mm) = (F(px)_x_or_y * Sensor_width_or_length(mm) ) / n_pixels_x_or_y
F_mm = None
f_px_x = None
f_px_y = None

# Transformation from world coordinate frame to camera:
# Translation
T_x = None
T_y = None
T_z = None
T = [T_x, T_y, T_z]
# Rotation
roll = None
pitch = None
yaw = None
R = None # Rotation matrix: Do get_R_Mat([roll, pitch, yaw])


def get_P_Mat():
    K = np.asarray(get_K_Mat())
    R = np.asarray(get_R_Mat())
    T_t = np.asarray(T).T
    RT = np.concatenate(R,T, axis=0)
    fill = np.asarray([0, 0, 0, 1])
    RT = np.concatenate(RT, fill, axis=1)
    P = np.matmul(K, RT)
    return P

def get_R_Mat(angles):
    assert (angles is None or angles.any(None), "Angles are not initialized!")
    r = Rotation.from_euler("zyx", angles, degrees=True)
    new_rotation_matrix = r.as_matrix()
    return new_rotation_matrix

def get_K_Mat():
    assert(f_px_x is None or f_px_y is None, "Focal pixel lengths are not initialized!")
    assert(cx is None or cy is None, "Centers are not initialized!")
    K = np.asarray([[f_px_x, 0, cx],
                   [0, f_px_y, cy],
                   [0, 0, 1]])
    return K

def main():
    K = get_K_Mat()


if __name__ == "__main__":
    main()