# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
# Function to compute the rotation matrices from the quaternions
def compute_rotation_matrices(data_dict):
    q_0 = data_dict['euler_1']
    q_1 = data_dict['euler_2']
    q_2 = data_dict['euler_3']
    q_3 = data_dict['euler_4']
    
    rotation_matrices = []
    
    for i in range(len(q_0)):
        
        # Compute the rotation matrix

        R = np.array([
            [1-2*(q_2[i]**2+q_3[i]**2), 2*(q_1[i]*q_2[i] - q_0[i]*q_3[i]), 2*(q_1[i]*q_3[i] + q_0[i]*q_2[i])],
            [2*(q_1[i]*q_2[i] + q_0[i]*q_3[i]), 1-2*(q_1[i]**2+q_3[i]**2), 2*(q_2[i]*q_3[i] - q_0[i]*q_1[i])],
            [2*(q_1[i]*q_3[i] - q_0[i]*q_2[i]), 2*(q_2[i]*q_3[i] + q_0[i]*q_1[i]), 1-2*(q_1[i]**2+q_2[i]**2)]
            ])    

        rotation_matrices.append(R)
    
    return rotation_matrices

# %%
# Function to multiply two quaternions
def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]

# Function to compute the conjugate of a quaternion
def quaternion_conjugate(q):
    w, x, y, z = q
    return [w, -x, -y, -z]

# Function to compute angular velocity from quaternion and its derivative
def quaternion_to_angular_velocity(q, dq_dt):
    q_conj = quaternion_conjugate(q)
    omega_quat = quaternion_multiply(q_conj, dq_dt)
    return 2 * np.array(omega_quat[1:])  # Extract the vector part

def compute_angular_velocity(data_dict):
    time_steps = data_dict['time']
    angular_velocities = [np.array([0, 0, 0])]  # First entry is zero
    for i in range(len(time_steps) - 1):
        dt = time_steps[i + 1] - time_steps[i]

        # Extract quaternions at consecutive time steps
        q1 = np.array([data_dict['euler_1'][i], data_dict['euler_2'][i], data_dict['euler_3'][i], data_dict['euler_4'][i]])
        q2 = np.array([data_dict['euler_1'][i + 1], data_dict['euler_2'][i + 1], data_dict['euler_3'][i + 1], data_dict['euler_4'][i + 1]])

        # Compute the quaternion derivative
        dq_dt = (q2 - q1) / dt

        # Compute angular velocity in the local frame
        omega = quaternion_to_angular_velocity(q1, dq_dt)
        angular_velocities.append(omega)

    data_dict['omega'] = np.array(angular_velocities)

# Function to compute angular accelerations
def compute_angular_acceleration(data_dict):
    time_steps = data_dict['time']
    angular_accelerations = [np.array([0.2073051, -3.270337, 0])]
    angular_velocities = data_dict['omega']
    for i in range(len(angular_velocities) - 1):
        dt = time_steps[i + 1] - time_steps[i]

        omega1 = angular_velocities[i]
        omega2 = angular_velocities[i + 1]

        # Compute angular acceleration
        alpha = (omega2 - omega1) / dt
        angular_accelerations.append(alpha)

    data_dict['omega_dot'] = np.array(angular_accelerations)

# %%
def tilde(V):
    V_tilde = np.array([
        [    0, -V[2],  V[1]],
        [ V[2],     0, -V[0]],
        [-V[1],  V[0],     0]
        ])
    return V_tilde

# %%
def omega_tilde(data_dict):
    omega = data_dict['omega']
    
    omegas_tilde = []
    
    for o in omega:
        omegas_tilde.append(tilde(o))
    
    return np.array(omegas_tilde)

# %%
def omega_dot_tilde(data_dict):
    omega = data_dict['omega']
    
    omegas_dot_tilde = []
    
    for o in omega:
        omegas_dot_tilde.append(tilde(o))
    
    return np.array(omegas_dot_tilde)

# %%
def mat_a(data_dict):
    omegas_tilde = data_dict['omega_tilde']
    omegas_dot_tilde = data_dict['omega_dot_tilde']
    
    mat_a = []
    
    for ot, odt in zip(omegas_tilde, omegas_dot_tilde):
        mat_a.append(ot @ ot + odt)
    
    return np.array(mat_a)

# %%
def g_local(data_dict, g_val):

    R = data_dict['rotation_matrices']
    g = np.array([0, -g_val, 0])
    
    g_local = []

    for r in R:
        g_local.append(np.dot(r, g))

    return np.array(g_local)

# %%
def mat_b(data_dict):
    a_x = data_dict['accel_x']
    a_y = data_dict['accel_y']
    a_z = data_dict['accel_z']

    R = data_dict['rotation_matrices']
    g = data_dict['g_local']

    mat_b = []

    for i in range(len(a_x)):
        X_ddot = np.array([a_x[i], a_y[i], a_z[i]])
        X_ddot_tilde = tilde(X_ddot)
        g_tilde = tilde(g[i].flatten())
        mat_b.append(X_ddot_tilde.T + g_tilde.T)
    
    return np.array(mat_b)

# %%
def mat_n(data_dict):
    omega = data_dict['omega']
    omega_dot = data_dict['omega_dot']

    mat_n = []

    for i in range(len(omega)):
        omega_x, omega_y, omega_z = omega[i]
        omega_dot_x, omega_dot_y, omega_dot_z = omega_dot[i]

        n = np.array([
            [omega_dot_x, -omega_y * omega_z, omega_y * omega_z, omega_dot_y - omega_x * omega_z, omega_dot_z + omega_x * omega_y, omega_y**2 - omega_z**2],
            [omega_x * omega_z, omega_dot_y, omega_x * omega_z, omega_dot_x + omega_y * omega_z, omega_z**2 - omega_x**2, omega_dot_z - omega_x * omega_y],
            [-omega_x * omega_y, omega_x * omega_y, omega_dot_z, omega_x**2 - omega_y**2, omega_dot_x - omega_y * omega_z, omega_dot_y + omega_x * omega_z]
        ])
        mat_n.append(n)

    return np.array(mat_n)

# %%
def CG(data_dict, m):
    force_x = data_dict['j4_fx']
    force_y = data_dict['j4_fy']
    force_z = data_dict['j4_fz']
    a_x = data_dict['accel_x']
    a_y = data_dict['accel_y']
    a_z = data_dict['accel_z']
    g = data_dict['g_local']
    
    a = data_dict['mat_a']
    F = []
    A = []
    
    for i in range(len(force_x)):
        f = np.array([force_x[i]/m, force_y[i]/m, force_z[i]/m]) - np.array([a_x[i], a_y[i], a_z[i]]) - g[i]
        F.append(f)
        A.append(a[i])

    # Stack matrices for least squares solution
    A_stacked = np.vstack(A)
    F_stacked = np.hstack(F)

    # Solve for the center of gravity coordinates (x, y, z) in the least-squares sense
    center_of_gravity, residuals, rank, s = np.linalg.lstsq(A_stacked, F_stacked, rcond=None)

    # Output results
    x, y, z = center_of_gravity
    print(f"Center of Gravity: x={x}, y={y}, z={z}")
    return center_of_gravity

# %%
def compute_torques(data_dict, m, cog, l4, l5, l6, lf):
    j4_fx = data_dict['j4_fx']
    j4_fy = data_dict['j4_fy']
    j4_fz = data_dict['j4_fz']
    j5_fx = data_dict['j5_fx']
    j5_fy = data_dict['j5_fy']
    j5_fz = data_dict['j5_fz']
    j6_fx = data_dict['j6_fx']
    j6_fy = data_dict['j6_fy']
    j6_fz = data_dict['j6_fz']
    sf_fx = data_dict['sforce_x']
    sf_fy = data_dict['sforce_y']
    sf_fz = data_dict['sforce_z']
    g_local = data_dict['g_local']
    actual_cog = np.array([0.0, -0.5, 0.5])
    delta_cg = actual_cog - cog
    
    torques = []

    for i in range(len(j4_fx)):

        j4 = np.array([j4_fx[i], j4_fy[i], j4_fz[i]])
        j5 = np.array([j5_fx[i], j5_fy[i], j5_fz[i]])
        j6 = np.array([j6_fx[i], j6_fy[i], j6_fz[i]])
        sf = np.array([sf_fx[i], sf_fy[i], sf_fz[i]])
        g = g_local[i]
        
        torque = np.cross(l4, j4) + np.cross(l5, j5) + np.cross(l6, j6) + np.cross(lf, sf) + np.cross(delta_cg, m * g)
        
        torques.append(torque)
    
    return np.array(torques)

# %%
def rearrange_tensor(inertia_components):
    inertia_tensor = np.array([inertia_components[0], inertia_components[3], inertia_components[4],
                    inertia_components[3], inertia_components[1], inertia_components[5],
                    inertia_components[4], inertia_components[5], inertia_components[2]]).reshape(3, 3)
    return inertia_tensor

# %%
def inertia(data_dict):
    
    n = data_dict['mat_n']
    torques = data_dict['torques']

    N = []
    Torques = []

    for i in range(len(n)):
        N.append(n[i])
        Torques.append(torques[i].flatten())  # Flatten the torque array to ensure compatibility

    # Stack matrices for least squares solution
    N_stacked = np.vstack(N)
    T_stacked = np.vstack(Torques).flatten()  # Flatten the torque array to ensure compatibility

    # Solve for the inertia tensor components in the least-squares sense
    inertia_components, residuals, rank, s = np.linalg.lstsq(N_stacked, T_stacked[:, np.newaxis], rcond=None)

    # Output the components of the inertia tensor
    
    inertia_tensor = rearrange_tensor(inertia_components)
    
    print(f"Inertia Tensor:")
    print(inertia_tensor)
    
    return inertia_components, inertia_tensor

# %%
# Read the data from the file
data = pd.read_csv('cube_20250104_2.tab', sep='\s+', skiprows=1, header=0)

# Convert the DataFrame to a NumPy array
data = data.to_numpy()

# Define the keys corresponding to each column
keys = ['time', 'euler_1', 'euler_2', 'euler_3', 'euler_4', 'accel_x', 'accel_y', 'accel_z', 
        'omega_dot_x', 'omega_dot_y', 'omega_dot_z', 'omega_x', 'omega_y', 'omega_z', 
        'force_x', 'force_y', 'force_z', 'j4_fx', 'j4_fy', 'j4_fz', 
        'j5_fx', 'j5_fy', 'j5_fz', 'j6_fx', 'j6_fy', 'j6_fz', 'sforce_x', 'sforce_y', 'sforce_z']

# Create a dictionary mapping keys to columns
data_dict = {key: data[:, idx] for idx, key in enumerate(keys)}

# Compute the rotation matrices and add them to the data_dict
data_dict['rotation_matrices'] = compute_rotation_matrices(data_dict)

# Compute angular velocity and acceleration
compute_angular_velocity(data_dict)
compute_angular_acceleration(data_dict)

# Compute the skew matrices of omega and omega_dot and add them to the data_dict
data_dict['omega_tilde'] = omega_tilde(data_dict)
data_dict['omega_dot_tilde'] = omega_dot_tilde(data_dict)

# Compute the local acceleration due to gravity and add it to the data_dict
g = 9.80665
data_dict['g_local'] = g_local(data_dict, g)

# Compute the matrices a, b, and n and add them to the data_dict
data_dict['mat_a'] = mat_a(data_dict)
data_dict['mat_b'] = mat_b(data_dict)
data_dict['mat_n'] = mat_n(data_dict)

# Compute the location of center of gravity
m = 7801  # mass [kg]
cg = CG(data_dict, m)

# %%
cog = np.array([0.0, -0.5, 0.5])

# Compute the torques and add them to the data_dict
P4 = np.array([0, 0.5, 0]) # joint 4
P5 = np.array([-0.5, 0.5, 0.8660254038]) # joint 5
P6 = np.array([0.5, 0.5, 0.8660254038]) # joint 6
PF = np.array([0.5, -0.5, 0]) # force application point

l_g4 = P4 - cog # vector from center of gravity to joint 4
l_g5 = P5 - cog # vector from center of gravity to joint 5
l_g6 = P6 - cog # vector from center of gravity to joint 6
l_gF = PF - cog # vector from center of gravity to force application point

data_dict['torques'] = compute_torques(data_dict, m, cog, l_g4, l_g5, l_g6, l_gF)

# Compute the inertia tensor
inertia_components, inertia_tensor = inertia(data_dict)

# %%
# Compute the torques and add them to the data_dict
P4 = np.array([0, 0.5, 0]) # joint 4 (reference point)
P5 = np.array([-0.5, 0.5, 0.8660254038]) # joint 5
P6 = np.array([0.5, 0.5, 0.8660254038]) # joint 6
PF = np.array([0.5, -0.5, 0]) # force application point

# Compute the torques and add them to the data_dict
P4 = np.array([0, 0.5, 0]) # joint 4
P5 = np.array([-0.5, 0.5, 0.8660254038]) # joint 5
P6 = np.array([0.5, 0.5, 0.8660254038]) # joint 6
PF = np.array([0.5, -0.5, 0]) # force application point

l_g4 = P4 - cg # vector from center of gravity to joint 4
l_g5 = P5 - cg # vector from center of gravity to joint 5
l_g6 = P6 - cg # vector from center of gravity to joint 6
l_gF = PF - cg # vector from center of gravity to force application point

data_dict['torques'] = compute_torques(data_dict, m, cg, l_g4, l_g5, l_g6, l_gF)

# Compute the inertia tensor
inertia_components, inertia_tensor = inertia(data_dict)

# %%
# Read the data from the file
data = pd.read_csv('cube_20250104_3.tab', sep='\s+', skiprows=1, header=0)

# Convert the DataFrame to a NumPy array
data = data.to_numpy()

# Define the keys corresponding to each column
keys = ['time', 'euler_1', 'euler_2', 'euler_3', 'euler_4', 'accel_x', 'accel_y', 'accel_z', 
        'omega_dot_x', 'omega_dot_y', 'omega_dot_z', 'omega_x', 'omega_y', 'omega_z', 
        'force_x', 'force_y', 'force_z', 'j4_fx', 'j4_fy', 'j4_fz', 
        'j5_fx', 'j5_fy', 'j5_fz', 'j6_fx', 'j6_fy', 'j6_fz', 'sforce_x', 'sforce_y', 'sforce_z']

# Create a dictionary mapping keys to columns
data_dict = {key: data[:, idx] for idx, key in enumerate(keys)}

# Compute the rotation matrices and add them to the data_dict
data_dict['rotation_matrices'] = compute_rotation_matrices(data_dict)

# Compute angular velocity and acceleration
compute_angular_velocity(data_dict)
compute_angular_acceleration(data_dict)

# Compute the skew matrices of omega and omega_dot and add them to the data_dict
data_dict['omega_tilde'] = omega_tilde(data_dict)
data_dict['omega_dot_tilde'] = omega_dot_tilde(data_dict)

# Compute the local acceleration due to gravity and add it to the data_dict
g = 9.80665
data_dict['g_local'] = g_local(data_dict, g)

# Compute the matrices a, b, and n and add them to the data_dict
data_dict['mat_a'] = mat_a(data_dict)
data_dict['mat_b'] = mat_b(data_dict)
data_dict['mat_n'] = mat_n(data_dict)

# Compute the location of center of gravity
m = 7801  # mass [kg]
cg = CG(data_dict, m)

# Compute the torques and add them to the data_dict
P4 = np.array([0, 0.5, 0]) # joint 4 (reference point)
P5 = np.array([-0.5, 0.5, 0.8660254038]) # joint 5
P6 = np.array([0.5, 0.5, 0.8660254038]) # joint 6
PF = np.array([0.5, -0.5, 0]) # force application point

# Compute the torques and add them to the data_dict
P4 = np.array([0, 0.5, 0]) # joint 4
P5 = np.array([-0.5, 0.5, 0.8660254038]) # joint 5
P6 = np.array([0.5, 0.5, 0.8660254038]) # joint 6
PF = np.array([0.5, -0.5, 0]) # force application point

l_g4 = P4 - cg # vector from center of gravity to joint 4
l_g5 = P5 - cg # vector from center of gravity to joint 5
l_g6 = P6 - cg # vector from center of gravity to joint 6
l_gF = PF - cg # vector from center of gravity to force application point

data_dict['torques'] = compute_torques(data_dict, m, cg, l_g4, l_g5, l_g6, l_gF)

# Compute the inertia tensor
inertia_components, inertia_tensor = inertia(data_dict)


