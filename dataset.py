import torch.utils.data as data
import cv2
from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import pandas as pd
import torch
import torchvision.transforms.functional as tF
from tools.tools import *


def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def min_frame(self):
        return int(self._data[2])

    @property
    def max_frame(self):
        return int(self._data[3])


class TSNDataSet(data.Dataset):
    def __init__(self, mode,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None, transform_1=None,
                 force_grayscale=False, random_shift=True, test_mode=False, context=False,
                 normalize_stgcn=True,
                 centralize_stgcn=False,
                 random_choose_stgcn=False,
                 random_move_stgcn=False,
                 random_shift_stgcn=False
                 ):

# ######################################################################
        self.random_choose_stgcn = random_choose_stgcn
        self.random_move_stgcn = random_move_stgcn
        self.random_shift_stgcn = random_shift_stgcn
        self.normalize_stgcn = normalize_stgcn
        self.centralize_stgcn = centralize_stgcn
        self.T = 297
        # max joint sequence length
        if mode == "train":
            self.max_x = np.load("/home/a/Desktop/NTUA-BEEU-eccv2020-master/misc/BOLD_train_max_x_joint.npy")
            self.max_y = np.load("/home/a/Desktop/NTUA-BEEU-eccv2020-master/misc/BOLD_train_max_y_joint.npy")
        elif mode == "val":
            self.max_x = np.load("/home/a/Desktop/NTUA-BEEU-eccv2020-master/misc/BOLD_val_max_x_joint.npy")
            self.max_y = np.load("/home/a/Desktop/NTUA-BEEU-eccv2020-master/misc/BOLD_val_max_y_joint.npy")
        elif mode == 'test':
            self.max_x = np.load("/home/a/Desktop/NTUA-BEEU-eccv2020-master/misc/BOLD_test_max_x_joint.npy")
            self.max_y = np.load("/home/a/Desktop/NTUA-BEEU-eccv2020-master/misc/BOLD_test_max_y_joint.npy")
# ######################################################################
        self.num_segments = num_segments
        self.new_length = new_length
        self.new_length_1 = 5
        self.modality = modality
        self.modality_1 = 'flow'
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.transform_1 = transform_1
        self.random_shift = random_shift
        self.test_mode = test_mode

        self.bold_path = "/home/a/Documents/BOLD_public"

        self.context = context
        self.context_1 = False

        self.categorical_emotions = ["Peace", "Affection", "Esteem", "Anticipation", "Engagement", "Confidence",
                                     "Happiness",
                                     "Pleasure", "Excitement", "Surprise", "Sympathy", "Doubt/Confusion", "Disconnect",
                                     "Fatigue", "Embarrassment", "Yearning", "Disapproval", "Aversion", "Annoyance",
                                     "Anger",
                                     "Sensitivity", "Sadness", "Disquietment", "Fear", "Pain", "Suffering"]

        self.continuous_emotions = ["Valence", "Arousal", "Dominance"]

        self.attributes = ["Gender", "Age", "Ethnicity"]

        header = ["video", "person_id", "min_frame",
                  "max_frame"] + self.categorical_emotions + self.continuous_emotions + self.attributes + [
                     "annotation_confidence"]

        # self.df = pd.read_csv(os.path.join(self.bold_path, "annotations/{}_extra.csv".format(mode)))
        self.df = pd.read_csv(os.path.join(self.bold_path, "annotations/{}.csv".format(mode)), names=header)
        self.df["joints_path"] = self.df["video"].apply(rreplace, args=[".mp4", ".npy", 1])

        self.video_list = self.df["video"]
        self.mode = mode

        self.embeddings = np.load("glove_840B_embeddings.npy")

    def _load_joints(self, directory, idx, index):

        joints = self.joints(index)

        poi_joints = joints[joints[:, 0] + 1 == idx]
        sample = self.df.iloc[index]
        poi_joints = poi_joints[(poi_joints[:, 1] == sample["person_id"]), 2:]

        if poi_joints.size == 0:
            poi_joints = np.zeros((18, 3))
        else:
            poi_joints = poi_joints.reshape((18, 3))

        poi_joints[poi_joints[:, 2] < 0.1] = np.nan
        poi_joints[np.isnan(poi_joints[:, 2])] = np.nan

        return poi_joints

    def get_context(self, image, joints, format="cv2"):
        joints = joints.reshape((18, 3))
        joints[joints[:, 2] < 0.1] = np.nan
        joints[np.isnan(joints[:, 2])] = np.nan

        joint_min_x = int(round(np.nanmin(joints[:, 0])))
        joint_min_y = int(round(np.nanmin(joints[:, 1])))

        joint_max_x = int(round(np.nanmax(joints[:, 0])))
        joint_max_y = int(round(np.nanmax(joints[:, 1])))

        expand_x = int(round(10 / 100 * (joint_max_x - joint_min_x)))
        expand_y = int(round(10 / 100 * (joint_max_y - joint_min_y)))

        if format == "cv2":
            image[max(0, joint_min_x - expand_x):min(joint_max_x + expand_x, image.shape[1])] = [0, 0, 0]
        elif format == "PIL":
            bottom = min(joint_max_y + expand_y, image.height)
            right = min(joint_max_x + expand_x, image.width)
            top = max(0, joint_min_y - expand_y)
            left = max(0, joint_min_x - expand_x)
            image = np.array(image)
            if len(image.shape) == 3:
                image[top:bottom, left:right] = [0, 0, 0]
            else:
                image[top:bottom, left:right] = np.min(image)
            return Image.fromarray(image)

    def get_bounding_box(self, image, joints, format="cv2"):
        joints = joints.reshape((18, 3))
        joints[joints[:, 2] < 0.1] = np.nan
        joints[np.isnan(joints[:, 2])] = np.nan

        joint_min_x = int(round(np.nanmin(joints[:, 0])))
        joint_min_y = int(round(np.nanmin(joints[:, 1])))

        joint_max_x = int(round(np.nanmax(joints[:, 0])))
        joint_max_y = int(round(np.nanmax(joints[:, 1])))

        expand_x = int(round(100 / 100 * (joint_max_x - joint_min_x)))
        expand_y = int(round(100 / 100 * (joint_max_y - joint_min_y)))

        if format == "cv2":
            return image[max(0, joint_min_y - expand_y):min(joint_max_y + expand_y, image.shape[0]),
                   max(0, joint_min_x - expand_x):min(joint_max_x + expand_x, image.shape[1])]
        elif format == "PIL":
            bottom = min(joint_max_y + expand_y, image.height)
            right = min(joint_max_x + expand_x, image.width)
            top = max(0, joint_min_y - expand_y)
            left = max(0, joint_min_x - expand_x)
            return tF.crop(image, top, left, bottom - top, right - left)

    def joints(self, index):
        sample = self.df.iloc[index]

        joints_path = os.path.join(self.bold_path, "joints", sample["joints_path"])

        joints18 = np.load(joints_path)
        joints18[:, 0] -= joints18[0, 0]

        return joints18

    def _load_image(self, directory, idx, index, mode="body"):
        joints = self.joints(index)
        # joints是从0开始数的，但是我们这里的index是从1开始数的，所以需要加1，把对应的那一帧的所有信息全部取出来
        poi_joints = joints[joints[:, 0] + 1 == idx]
        sample = self.df.iloc[index]
        poi_joints = poi_joints[(poi_joints[:, 1] == sample["person_id"]), 2:] # 把对应的那个人的肢体信息提取出来
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            image_path = os.path.join(directory, self.image_tmpl.format(idx))
            frame = Image.open(image_path).convert("RGB")

            if mode == "context":
                if poi_joints.size == 0:
                    return [frame]
                context = self.get_context(frame, poi_joints, format="PIL")
                return [context]

            if poi_joints.size == 0:
                body = frame
                pass  # just do the whole frame
            else:
                body = self.get_bounding_box(frame, poi_joints, format="PIL")

                if body.size == 0:
                    print(poi_joints)
                    body = frame

            return [body]

    def _load_image_1(self, directory, idx, index, mode="body"):
        joints = self.joints(index)
        # joints是从0开始数的，但是我们这里的index是从1开始数的，所以需要加1，把对应的那一帧的所有信息全部取出来
        poi_joints = joints[joints[:, 0] + 1 == idx]
        sample = self.df.iloc[index]
        poi_joints = poi_joints[(poi_joints[:, 1] == sample["person_id"]), 2:]
        # return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        if self.modality_1 == 'flow':
            frame_x = Image.open(os.path.join(directory, "{}_{:05d}.jpg".format('flow_x', idx))).convert('L')
            frame_y = Image.open(os.path.join(directory, "{}_{:05d}.jpg".format('flow_y', idx))).convert('L')
            # frame = cv2.imread(os.path.join(directory, 'img_{:05d}.jpg'.format(idx)))
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if mode == "context":
                if poi_joints.size == 0:
                    return [frame_x, frame_y]
                context_x = self.get_context(frame_x, poi_joints, format="PIL")
                context_y = self.get_context(frame_y, poi_joints, format="PIL")
                return [context_x, context_y]

            if poi_joints.size == 0:
                body_x = frame_x
                body_y = frame_y
                pass  # just do the whole frame
            else:
                body_x = self.get_bounding_box(frame_x, poi_joints, format="PIL")
                body_y = self.get_bounding_box(frame_y, poi_joints, format="PIL")

                if body_x.size == 0:
                    body_x = frame_x
                    body_y = frame_y

            return [body_x, body_y]

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """

        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                              size=self.num_segments)  # + (record.min_frame+1)
            # print(record.num_frames, record.min_frame, record.max_frame)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _sample_indices_1(self, record):
        """

        :param record: VideoRecord
        :return: list
        """

        average_duration = (record.num_frames - self.new_length_1 + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                              size=self.num_segments)  # + (record.min_frame+1)
            # print(record.num_frames, record.min_frame, record.max_frame)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length_1 + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices_1(self, record):
        if record.num_frames > self.num_segments + self.new_length_1 - 1:
            tick = (record.num_frames - self.new_length_1 + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):

        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets + 1

    def _get_test_indices_1(self, record):

        tick = (record.num_frames - self.new_length_1 + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets + 1

    def __getitem__(self, index):
        sample = self.df.iloc[index]
        # 得到对应的那一行
        fname = os.path.join(self.bold_path, "videos", self.df.iloc[index]["video"])
        # '/home/a/Documents/BOLD_public/videos/003/N7baJsMszJ0.mp4/0567.mp4'
        capture = cv2.VideoCapture(fname)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

        capture.release()
        # '/home/a/Documents/BOLD_public/test_raw/N7baJsMszJ0.mp4/0567'
        record_path = os.path.join(self.bold_path, "test_raw", sample["video"][4:-4])

        record = VideoRecord([record_path, frame_count, sample["min_frame"], sample["max_frame"]])

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        if not self.test_mode:
            segment_indices_1 = self._sample_indices_1(record) if self.random_shift else self._get_val_indices_1(record)
        else:
            segment_indices_1 = self._get_test_indices_1(record)

        return self.get(record, segment_indices, segment_indices_1, index)

    def get(self, record, indices, indices_1, index):

        images = list()

        # print(indices)
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):

                seg_imgs = self._load_image(record.path, p, index, mode="body")
                # seg_imgs[0].show()
                images.extend(seg_imgs)

                if self.context:
                    seg_imgs = self._load_image(record.path, p, index, mode="context")
                    # seg_imgs[0].show()
                    images.extend(seg_imgs)

                if p < record.num_frames:
                    p += 1

        images_1 = list()

        for seg_ind in indices_1:
            p = int(seg_ind)
            for i in range(5):

                seg_imgs = self._load_image_1(record.path, p, index, mode="body")
                # seg_imgs[0].show()
                images_1.extend(seg_imgs)

                if self.context_1:
                    seg_imgs = self._load_image_1(record.path, p, index, mode="context")
                    # seg_imgs[0].show()
                    images_1.extend(seg_imgs)

                if p < record.num_frames:
                    p += 1

        joints = list()
        for ind in range(1, record.num_frames):
            p = int(ind)
            j = self._load_joints(record.path, p, index)
            j[np.isnan(j)] = 0
            if self.normalize_stgcn:
                j[:, 0] = j[:, 0] / float(self.max_x[index])
                j[:, 1] = j[:, 1] / float(self.max_y[index])
            if self.centralize_stgcn:
                j[:, 0] = j[:, 0] - 0.5
                j[:, 1] = j[:, 1] - 0.5
            joints.append(np.transpose(j))

        joints = np.stack(joints, axis=1)
        joints = np.array(np.expand_dims(joints, axis=-1))

        if self.random_shift_stgcn:
            joints = random_shift_stgcn(joints)
        if self.random_choose_stgcn:
            joints = random_choose_stgcn(joints, self.T)
        else:
            joints = auto_padding_stgcn(joints, self.T, random_pad=(self.mode == "train"))
        if self.random_move_stgcn:
            joints = random_move_stgcn(joints)

        if not self.test_mode:
            categorical = self.df.iloc[index][self.categorical_emotions]
            continuous = self.df.iloc[index][self.continuous_emotions]
            continuous = continuous / 10.0  # normalize to 0 - 1

            if self.transform is None:
                process_data = images
                process_data_1 = images_1
            else:
                process_data = self.transform(images)
                process_data_1 = self.transform_1(images_1)

            return torch.tensor(joints).float(), process_data, process_data_1, torch.tensor(self.embeddings).float(), torch.tensor(categorical).float(), torch.tensor(
                continuous).float(), self.df.iloc[index]["video"], index
        else:
            process_data = self.transform(images)
            process_data_1 = self.transform_1(images_1)
            return torch.tensor(joints).float(), process_data, process_data_1, torch.tensor(self.embeddings).float()

    def __len__(self):
        return len(self.df)
