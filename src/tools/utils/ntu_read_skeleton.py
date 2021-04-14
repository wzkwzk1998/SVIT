import os
import numpy as np

def read_skeleton(file):
    """
    从文件中读取节点数据的信息
    :return:
    """
    with open(file, 'r') as f:
        skeleton_seq = {}
        skeleton_seq['numFrame'] = int(f.readline())            # 读取一个视频序列中有多少帧
        skeleton_seq['frameInfo'] = []
        for t in range(skeleton_seq['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())  # 该动作执行的人数
            frame_info['bodyInfo'] = []                # 骨架信息
            for m in range(frame_info['numBody']):
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k : float(v)
                    for k, v in zip(body_info_key,f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())               # 节点个数
                body_info['jointInfo'] = []                             # 节点信息
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k : float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_seq['frameInfo'].append(frame_info)
    return skeleton_seq



def read_xyz(file, max_body=2, num_joint=25):
    seq_info = read_skeleton(file)
    data = np.zeros((3, seq_info['numFrame'], num_joint, max_body))             # data shape is (channel, T, V, M)
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j,v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[:, n, j, m] = [v['x'], v['y'], v['z']]
    return data


if __name__ == '__main__':
    data_path = '/Users/wzk1998/data/NTU_RGB+D'
    test_skeleton = 'S014C002P037R002A050.skeleton'

    data = read_xyz(os.path.join(data_path, test_skeleton))
    print("data shape is {}".format(data.shape))
    print(data)


