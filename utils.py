import numpy as np
import glob
import pyemma as py
from hde import HDE, analysis
import mdtraj as md


def get_pairs(num_bp, add_inter=False):  #BSP
    '''return all intermolecular pairs for abasic seqs'''
    
    s1, atom_pairs = [], []
    s1 = [i*3 + 1 for i in range(num_bp)]
    s2 = [i*3 + num_bp*3 for i in range(num_bp)]
    
    # intermolecular
    for b1 in s1:
        for b2 in s2:
            atom_pairs.append((b1, b2))

    # intra molecular for strand 1
    for i, b1 in enumerate(s1):
        for b2 in s1[:i]:
            atom_pairs.append((b1, b2))
            
    # intra molecular for strand 2
    for i, b1 in enumerate(s2):
        for b2 in s2[:i]:
            atom_pairs.append((b1, b2))

    return atom_pairs


  
def get_all_pairs(num_bp, b_idx, strand=1):
    '''return all intermolecular pairs for abasic seqs'''
    
    intra_pairs1, intra_pairs2, inter_pairs = [], [], []
    
    # account for control
    if b_idx is None:
        s1 = [i*3 + 1 for i in range(num_bp)]
        s2 = [i*3 + num_bp*3 for i in range(num_bp)]
    else:
        # shift index down for missing bp
        s1 = [i*3 + 1 for i in range(b_idx-1)] + [i*3 for i in range(b_idx, num_bp)]
        s2 = [i*3 + num_bp*3 - 1 for i in range(num_bp)]
        
    for b1 in s1:
        for b2 in s2:
            inter_pairs.append((b1, b2))
            
    for i, b1 in enumerate(s1):
        for j, b2 in enumerate(s1):
            if i==j: break
            intra_pairs1.append((b1, b2))
            
    for i, b1 in enumerate(s2):
        for j, b2 in enumerate(s2):
            if i==j: break
            intra_pairs2.append((b1, b2))
            
    return intra_pairs1, intra_pairs2, inter_pairs


def unwrap_coords(xyz, box_L=4.2757, ca_idx=16, cb_idx=45):
    '''wrap into specified box size
       box_L is set for a standard 11-mer and should be adjusted accordingly
       ca and cb indexes represent beads close to the center of each strand
       they should also be adjusted but do not need to be exactly in the middle'''
    
    frame_list = []
    for i in range(len(xyz)):
        
        #print('new frame')

        ## check for wrapping in trajs
        strand_a = xyz[i, :-32]
        strand_b = xyz[i, -32:]

        ## center on two arbitrary (central) base pairs
        ca_mean = xyz[i, ca_idx]
        cb_mean = xyz[i, cb_idx]
        
        # save initial strand dist
        diff_mean = ca_mean - cb_mean
        diff_total = diff_mean + box_L*2*(diff_mean < -box_L) - box_L*2*(diff_mean > box_L) 

        # subtract means of central base xyz
        strand_a_norm = strand_a - ca_mean + box_L
        strand_b_norm = strand_b - cb_mean + box_L
        #print(strand_a_norm[1::3])
        
        # calculate box shifts seperatley to prevent editing arraray
        strand_a_shift = box_L*2 * (strand_a_norm // (box_L*2))
        strand_b_shift = box_L*2 * (strand_b_norm // (box_L*2))
        #print(strand_a_shift[1::3])

        # connect all strands in same box
        strand_a_norm -= strand_a_shift
        strand_b_norm -= strand_b_shift
        #print(strand_a_norm[1::3])

        # once connected, add back in mean between the shifts
        strand_b_norm -= diff_total # this should be correct
    
        # now center all coordinates togther
        full_mean = (np.mean(strand_a_norm, axis=0) + np.mean(strand_b_norm, axis=0))/2
        
        strand_a_norm -= full_mean
        strand_b_norm -= full_mean
    
        xyz_fixed = np.append(strand_a_norm, strand_b_norm, axis=0)
        frame_list.append(xyz_fixed)
    
    print(np.shape(frame_list))
    return np.array(frame_list)

def save_new_xyz(name, save_dir, xyz):
    '''save synth xyzs to load as mdtraj objs'''

    n_atoms = np.shape(xyz)[1]
    with open(f'{save_dir}/{name}.xyz', 'w') as f:
        for i in range(xyz.shape[0]):
            f.write('%d\n' % n_atoms)
            f.write('\n')
            for k in range(n_atoms):
                # nm to Angstroms for xyz write
                f.write('%3s%17.9f%17.9f%17.9f\n' % 
                        ('C', xyz[i][k][0]*10, xyz[i][k][1]*10, xyz[i][k][2]*10) ) 
                
def reformat_xyz(xyz, ref_psf, ref_traj, ref_idxs=None, save_xyz=False, save_dir='./', save_name='test'):
    '''input xyz and return unwrapped, connected, and superposed xyz'''
    
    # wrap and save for visualization
    xyz_unwrap = unwrap_coords(xyz) 
    print('unwrapped coords')

    # need to save an intermediate xyz
    save_new_xyz('temp', save_dir, xyz_unwrap)
    print('saved temp xyz')
    
    # load into new traj and superpose 
    temp_load = f'{save_dir}/temp.xyz'
    traj_unwrap = md.load(temp_load, top=ref_psf)
    traj_unwrap.superpose(ref_traj, atom_indices=ref_idxs)
    xyz = traj_unwrap.xyz.reshape(-1, traj_unwrap.n_atoms*3)
    print('superposed')
    
    # save an output 
    if save_xyz:
        save_new_xyz(save_name, save_dir, traj_unwrap.xyz)
    
    return xyz


def fit_SRV(traj_list, dim, max_epochs, lag, verbose=True,
            batch_size=50000, lrate=0.01, val_split=0.0001):
    
    '''instantiate and fit SRV object, return the object'''
    
    SRV = HDE(
        np.shape(traj_list)[-1], 
        n_components=dim, 
        validation_split=val_split, 
        n_epochs=max_epochs, 
        lag_time=lag, 
        batch_size=batch_size, #500000
        #callbacks=calls,  # comment out for consistet training time
        learning_rate=lrate, 
        batch_normalization=True,
        latent_space_noise=0.0,
        verbose=verbose
    )
    
    SRV = SRV.fit(traj_list)
    return SRV

# transformation info
from scipy.spatial.transform import Rotation as R
from scipy.linalg import orthogonal_procrustes

def get_com_pros(xyz, xyz_ref, strand=None):
    '''Find difference in xyz center or mass comapred to reference'''
    
    # get com difference in xyz
    com_ref = np.mean(xyz_ref, axis=0)
    com = np.mean(xyz, axis=0)
    com_diff = com - com_ref
    
    # calculate relative proscutes
    xyz_ref_m = xyz_ref - com_ref
    xyz_m = xyz - com
    rot, _ = orthogonal_procrustes(xyz_m, xyz_ref_m)
    pros_diff = R.from_matrix(rot).as_euler('xyz')
    
    # should be no reflections 
    if np.linalg.det(rot.T) < 0:
        print ("Reflection detected")
    
    return np.concatenate((com_diff, pros_diff), axis=0)

def apply_com_pros(xyz, com_pros):
    '''Transform xyz according to center of mass and euler angles
    This function should work from the prop and be independent of reference info'''
    
    trans = com_pros[:3]
    
    if len(com_pros)==6:
        rot = R.from_euler('xyz', com_pros[3:]).as_matrix()
    elif len(com_pros)==7:
        rot = R.from_quat(com_pros[3:]).as_matrix()
    
    #com = np.mean(xyz, axis=0) # need com?
    
    # apply rotation then translation
    xyz = (xyz).dot(rot.T) - trans
    
    return xyz