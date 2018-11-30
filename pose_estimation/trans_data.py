import numpy as np
import os


def text_save(file, data):
    # file = open(filename,'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','') 
        s = s.replace("'",'').replace(',','') + ' '
        file.write(s)
    # file.close()

def split_4x3(inputNPY,outputTXT):
    trajectory = np.load(inputNPY)
    fid = open(outputTXT,'w')
    for index in range(0,len(trajectory)-2, trajectory.shape[1]-1):
        T_Cprev_C = trajectory[index][2]
        T_Cprev_C_f = np.ravel(T_Cprev_C)
        T_Cprev_C_txt = T_Cprev_C_f.tolist()
        # fid.write(T_Cprev_C_txt)
        text_save(fid,T_Cprev_C_txt)
        fid.write('\n')

    fid.close()
    tmp = 1

def relative2absolute(inputREl,outputABS,scale=1):
    pose_list = np.loadtxt(inputREl, dtype=np.float64)
    pose_list = pose_list.reshape((-1,3,4))
    poses_O_array = np.zeros((len(pose_list) + 1, 3, 4), dtype=np.float64)

    for index in range(len(poses_O_array)):
        if index == 0:
            poses_O_array[index][:,:3] = np.eye(3)
        else:
            poses_O_array[index][:,:3] = poses_O_array[index-1][:,:3] @ pose_list[index-1][:,:3]
            poses_O_array[index][:,-1] = poses_O_array[index-1][:,:3] @ pose_list[index-1][:,-1] * scale + poses_O_array[index-1][:,-1]
    
    fid = open(outputABS,'w')
    for index in range(0,len(poses_O_array)):
        T_O_C_f = np.ravel(poses_O_array[index])
        T_O_C_txt = T_O_C_f.tolist()
        text_save(fid,T_O_C_txt)
        fid.write('\n')

    fid.close()



if __name__ == '__main__':
    inputNPY = 'predictions.npy'
    outputTXT = 'Cpre_C_trajectory.txt'
    outputABS = 'O_C_trajectory.txt'
    split_4x3(inputNPY,outputTXT,)
    relative2absolute(outputTXT,outputABS,scale=46.37)

