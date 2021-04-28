import torch
import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle

sys.path.append(".")
from feeder import feeder
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import axes3d

data_path = "/Users/wzk1998/data/NTU-RGB-D/xview/eval_data.npy"
label_path = '/Users/wzk1998/data/NTU-RGB-D/xview/eval_label.pkl'
video_id = 'S001C001P001R001A020'


actions = {
    1: "drink water",
    2: "eat meal/snack",
    3: "brushing teeth",
    4: "brushing hair",
    5: "drop",
    6: "pickup",
    7: "throw",
    8: "sitting down",
    9: "standing up (from sitting position)",
    10: "clapping",
    11: "reading",
    12: "writing",
    13: "tear up paper",
    14: "wear jacket",
    15: "take off jacket",
    16: "wear a shoe",
    17: "take off a shoe",
    18: "wear on glasses",
    19: "take off glasses",
    20: "put on a hat/cap",
    21: "take off a hat/cap",
    22: "cheer up",
    23: "hand waving",
    24: "kicking something",
    25: "reach into pocket",
    26: "hopping (one foot jumping)",
    27: "jump up",
    28: "make a phone call/answer phone",
    29: "playing with phone/tablet",
    30: "typing on a keyboard",
    31: "pointing to something with finger",
    32: "taking a selfie",
    33: "check time (from watch)",
    34: "rub two hands together",
    35: "nod head/bow",
    36: "shake head",
    37: "wipe face",
    38: "salute",
    39: "put the palms together",
    40: "cross hands in front (say stop)",
    41: "sneeze/cough",
    42: "staggering",
    43: "falling",
    44: "touch head (headache)",
    45: "touch chest (stomachache/heart pain)",
    46: "touch back (backache)",
    47: "touch neck (neckache)",
    48: "nausea or vomiting condition",
    49: "use a fan (with hand or paper)/feeling warm",
    50: "punching/slapping other person",
    51: "kicking other person",
    52: "pushing other person",
    53: "pat on back of other person",
    54: "point finger at the other person",
    55: "hugging other person",
    56: "giving something to other person",
    57: "touch other person's pocket",
    58: "handshaking",
    59: "walking towards each other",
    60: "walking apart from each other",
    61: "put on headphone",
    62: "take off headphone",
    63: "shoot at the basket",
    64: "bounce ball",
    65: "tennis bat swing",
    66: "juggling table tennis balls",
    67: "hush (quite)",
    68: "flick hair",
    69: "thumb up",
    70: "thumb down",
    71: "make ok sign",
    72: "make victory sign",
    73: "staple book",
    74: "counting money",
    75: "cutting nails",
    76: "cutting paper (using scissors)",
    77: "snapping fingers",
    78: "open bottle",
    79: "sniff (smell)",
    80: "squat down",
    81: "toss a coin",
    82: "fold paper",
    83: "ball up paper",
    84: "play magic cube",
    85: "apply cream on face",
    86: "apply cream on hand back",
    87: "put on bag",
    88: "take off bag",
    89: "put something into a bag",
    90: "take something out of a bag",
    91: "open a box",
    92: "move heavy objects",
    93: "shake fist",
    94: "throw up cap/hat",
    95: "hands up (both hands)",
    96: "cross arms",
    97: "arm circles",
    98: "arm swings",
    99: "running on the spot",
    100: "butt kicks (kick backward)",
    101: "cross toe touch",
    102: "side kick",
    103: "yawn",
    104: "stretch oneself",
    105: "blow nose",
    106: "hit other person with something",
    107: "wield knife towards other person",
    108: "knock over other person (hit with body)",
    109: "grab other person’s stuff",
    110: "shoot at other person with a gun",
    111: "step on foot",
    112: "high-five",
    113: "cheers and drink",
    114: "carry something with other person",
    115: "take a photo of other person",
    116: "follow other person",
    117: "whisper in other person’s ear",
    118: "exchange things with other person",
    119: "support somebody with hand",
    120: "finger-guessing game (playing rock-paper-scissors)",
}

# ntu-rgb-d skeleton
skeleton = np.array([(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                     (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                     (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                     (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                     (22, 23), (23, 8), (24, 25), (25, 12)]) - 1

loader = torch.utils.data.DataLoader(dataset=feeder.Feeder(data_path=data_path, label_path=label_path),
                                     batch_size=64,
                                     shuffle=False,
                                     num_workers=2)

sample_name = loader.dataset.sample_name
sample_id = [name.split('.')[0] for name in sample_name]
sample_index = sample_id.index(video_id)
data, label = loader.dataset[sample_index]
data = data.reshape((1,) + data.shape)

with open(label_path, 'rb') as fp:
    sample_name, sample_label = pickle.load(fp)

sample_label = np.array(sample_label)
sample_label_index = sample_label[sample_index]

N, C, T, V, M = data.shape

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


def init():
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)


def update(index):
    ax.clear()
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    for sk in skeleton:
        x = np.array(data[0, 0, index, :, 0])
        y = np.array(data[0, 1, index, :, 0])
        z = np.array(data[0, 2, index, :, 0])

        x_line = np.array([x[sk[0]], x[sk[1]]])
        y_line = np.array([y[sk[0]], y[sk[1]]])
        z_line = np.array([z[sk[0]], z[sk[1]]])

        ax.plot(x_line, y_line, z_line, color='blue')

    action_class = sample_label_index + 1
    action_name = actions[action_class]
    plt.title('Skeleton Frame #{} of 300 \n (Action {}: {})'.format(index,
                                                                              action_class, action_name))


ani = FuncAnimation(fig, update, frames=T, repeat=True, init_func=init, interval=100)
plt.show()
