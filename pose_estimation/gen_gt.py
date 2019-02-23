import numpy as np 


def compute_pose_error(gt, pred):
    RE = 0
    snippet_length = gt.shape[0]
    scale_factor = np.sum(gt[:,:,:,-1] * pred[:,:,:,-1])/np.sum(pred[:,:,:,-1] ** 2)
    ATE = np.linalg.norm((gt[:,:,-1] - scale_factor * pred[:,:,-1]).reshape(-1))
    for gt_pose, pred_pose in zip(gt, pred):
        # Residual matrix to which we compute angle's sin and cos
        R = gt_pose[:,:3] @ np.linalg.inv(pred_pose[:,:3])
        s = np.linalg.norm([R[0,1]-R[1,0],
                            R[1,2]-R[2,1],
                            R[0,2]-R[2,0]])
        c = np.trace(R) - 1
        # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
        RE += np.arctan2(s,c)

    return ATE/snippet_length, RE/snippet_length


def test_C_O_Cpre():
    pose_list = np.loadtxt('00.txt', dtype=np.float64)
    pose_list = pose_list.reshape((4541,3,4))
    
    A = range(0,4539)
    sample_list = []
    for i in A:
        tmp = [i,i+1,i+2]
        sample_list.append(tmp)
    
    count = 0
    poses_array = np.zeros((4541, 3, 3, 4), dtype=np.float64)
    for snippet_indices in sample_list:
        poses_ = np.stack(pose_list[i] for i in snippet_indices)
        first_pose = poses_[0]
        poses_[:,:,-1] -= first_pose[:,-1]
        compensated_poses = np.linalg.inv(first_pose[:,:3]) @ poses_
        assert compensated_poses.shape == (3, 3, 4), print(compensated_poses.shape)
        poses_array[count] = np.array(compensated_poses)
        count += 1

    poses_array_gf = np.load('gt.npy')

    B = range(0,4539,2)
    poses_O_array = np.zeros((len(B), 3, 4), dtype=np.float64)
    for index in range(len(B)):
        if index == 0:
            poses_O_array[index][:,:3] = np.eye(3)
        else:
            poses_O_array[index][:,:3] = poses_O_array[index-1][:,:3] @ poses_array_gf[B[index-1]][2][:,:3]
            poses_O_array[index][:,-1] = poses_O_array[index-1][:,:3] @ poses_array_gf[B[index-1]][2][:,-1] + poses_O_array[index-1][:,-1]

    tmp = 1


if __name__ == '__main__':
    gt = np.load('gt.npy')
    pred = np.load('predictions.npy')
    compute_pose_error(gt,pred)