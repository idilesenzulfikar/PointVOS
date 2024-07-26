"""
This script extract all frames of videos, all annotations, annotated frames of videos, and finally visualize annotations
Supported datasets Kinetics, Oops
"""

import glob
import json
import os.path
import shutil
from argparse import ArgumentParser

import cv2
import numpy as np
from PIL import Image

from skimage.morphology import dilation, disk, square, rectangle

from dataset_scripts.vis_utils import annotate_instance, color_map

from dataset_scripts.keys import Dataset, dataset_to_str, str_to_dataset

BACKGROUND_CLASS_ID = 0
REJECTED_CLASS_ID = 1
ACCEPTED_CLASS_ID = 2
AMBIGUOUS_CLASS_ID = 255


def extract_video_frames(video_path: str, frames_path: str):
    """
    :param video_path: the path of the given video
    :param frames_path: the path where to extract frames for the given video
    """
    cap = cv2.VideoCapture(video_path)
    # get the FPS of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    # start the loop
    frame_id = 0
    while True:
        is_read, frame = cap.read()
        # break out of the loop if there are no frames to read
        if not is_read:
            break
        # get the duration by dividing the frame count by the FPS
        cv2.imwrite(os.path.join(frames_path, f"{frame_id:06d}.jpg"), frame)
        # drop the duration spot from the list, since this duration spot is already saved
        frame_id += 1


def extract_video_point_annotations(point_annotations: dict, frames_dir: str, annotations_dir: str,
                                    extract_ambiguous=False):
    """
    :param point_annotations: dict of lists, where each dict item is an annotated frame for the given video and list consists of annotations
    :param frames_dir: the path for the extracted frames for the given video
    :param annotations_dir: the path where to extract annotations for the given video
    :param extract_ambiguous: Flag for whether or not extract ambiguous annotations, default is False
    """
    for frame_id, points in point_annotations.items():
        frame_path = os.path.join(frames_dir, frame_id + '.jpg')
        assert os.path.exists(frame_path), "The frame {0} could not find in the video {}!".format(frame_id,
                                                                                                  frames_dir.split('/')[
                                                                                                      -1])
        annotation_path = os.path.join(annotations_dir, frame_id + '.png')

        image = np.array(Image.open(frame_path).convert('P'), dtype=np.uint8)  # H x W
        out_mask = np.zeros(image.shape[-2:]).astype(np.uint8)
        for p in points:
            x, y = p['x'], p['y']
            assert out_mask[y, x] == 0, "The coordinate is annotated twice!"
            if p['status'] == 'REJECTED':
                # bg annotation
                out_mask[y, x] = REJECTED_CLASS_ID
            elif p['status'] == 'ACCEPTED':
                #  fg annotation
                out_mask[y, x] = ACCEPTED_CLASS_ID
            elif p['status'] == 'AMBIGUOUS':
                if extract_ambiguous:
                    #  ambiguous annotation
                    out_mask[y, x] = AMBIGUOUS_CLASS_ID
            else:
                # a bug for annotation status
                raise ValueError('The point annotation status is invalid: {0}'.format(p['status']))

        mask_img = Image.fromarray(out_mask).convert('P')
        mask_img.putpalette(color_map().flatten().tolist())
        mask_img.save(annotation_path)


def extract_video_sparse_frames(frames_dir: str, annotations_dir: str, sparse_frames_dir: str):
    """
    :param frames_dir: the path for all frames for the given video
    :param annotations_dir: the path for the annotations for the given video
    :param sparse_frames_dir: the path where to extract only the annotated frames for the given video, basically it filters for the annotated frames
    """
    # get the sparse frames list from the extracted annotations
    # possibly it can be also get from json file as well...
    annotations_files = glob.glob(os.path.join(annotations_dir, '*.png'))
    annotations_files.sort()
    annotated_frames = [ann_file.split('/')[-1].split('.')[0] for ann_file in annotations_files]

    for frame_id in annotated_frames:
        frame_path = os.path.join(frames_dir, frame_id + '.jpg')
        filepath = os.path.join(sparse_frames_dir, frame_id + '.jpg')
        shutil.copy(frame_path, filepath)


def visualize_video_point_annotations(frames_dir: str, annotations_dir: str, visualizations_dir: str):
    """
    :param frames_dir: the path for the frames for the given video, you can use all frames or sparse frames.
    :param annotations_dir: the path for the annotations for the given video
    :param visualizations_dir: the path where to visualize the annotations on the frames for the given video
    """
    annotations_files = glob.glob(os.path.join(annotations_dir, '*.png'))
    annotations_files.sort()

    annotated_frames = [ann_file.split('/')[-1].split('.')[0] for ann_file in annotations_files]
    image_files = list(
        map(
            lambda x: os.path.join(frames_dir, x + '.jpg'),
            annotated_frames
        ))

    image_files.sort()

    for index, (f_img, f_ann) in enumerate(zip(image_files, annotations_files)):
        img = cv2.imread(f_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        point_mask = np.array(Image.open(f_ann).convert('P'), dtype=np.uint8)
        inst_range = np.unique(point_mask)  # number of instances should not be gotten from annotation values!!!
        inst_range = inst_range[inst_range != 0][:, None, None].astype(point_mask.dtype)

        one_hot = (point_mask[None] == inst_range).astype(np.uint8)
        one_hot = [dilation(_m, disk(5)) for _m in one_hot]

        cmap = {BACKGROUND_CLASS_ID: [0, 0, 0], REJECTED_CLASS_ID: [0, 0, 255], ACCEPTED_CLASS_ID: [18, 127, 15],
                AMBIGUOUS_CLASS_ID: [255, 255, 255]}

        inst_range = np.squeeze(np.squeeze(inst_range, axis=1), axis=1)
        colours = [cmap[inst] for inst in inst_range]

        if len(one_hot) == 0:
            annotated_image = img[:, :, ::-1]
        else:
            annotated_image = annotate_instance(img[:, :, ::-1], colours[0], mask=one_hot[0])
            for _i, _inst_mask in enumerate(one_hot[1:]):
                annotated_image = annotate_instance(annotated_image, colours[_i + 1], mask=_inst_mask)

        filepath = os.path.join(visualizations_dir, '{}.png'.format(f_img.split("/")[-1].split(".")[0]))
        cv2.imwrite(filepath, annotated_image)


if __name__ == '__main__':

    parser = ArgumentParser()

    # Which dataset? Supported datasets Kinetics/Oops
    parser.add_argument('--dataset', required=True, type=str, choices=[Dataset.kinetics.value, Dataset.oops.value],
                        help='The name of the dataset.')
    # You should already download videos, and save it into root
    parser.add_argument('--root', required=True, type=str,
                        help='The directory where the Kinetics/OOps annotations are saved...')
    # Extract train or val split of Oops, or train split of Kinetics
    parser.add_argument('--split', required=True, type=str, choices=['train', 'val'], help='Dataset split...')

    # What would you like to do? Extract frames, annotations, sparse frames or visualize annotations...
    parser.add_argument('--extract_frames', action='store_true', help='Extract all frames of videos...')
    parser.add_argument('--extract_annotations', action='store_true', help='Extract all frames of videos...')
    parser.add_argument('--extract_sparse_frames', action='store_true',
                        help='Extract sparse frames of videos, i.e. annotated frames...')
    parser.add_argument('--vis_annotations', action='store_true', help='Visualize annotations...')

    args = vars(parser.parse_args())
    assert os.path.exists(args['root']), "The dataset root does not exist!!"

    dataset = str_to_dataset(args['dataset'])
    root = os.path.join(args['root'], dataset_to_str(dataset))
    split = args['split']

    extract_frames = args['extract_frames']
    extract_annotations = args['extract_annotations']
    extract_sparse_frames = args['extract_sparse_frames']
    vis_annotations = args['vis_annotations']

    if (not extract_frames) and (not extract_annotations) and (not extract_sparse_frames) and (not vis_annotations):
        raise ValueError(
            "No boolen option is given, please provide one of extract_frames, extract_annotations, extract_sparse_frames, vis_annotations...")

    if not os.path.exists(root):
        os.makedirs(root)

    path = os.path.join(root, split)
    video_path = os.path.join(root, 'videos', split)

    if dataset == Dataset.kinetics:
        json_file = os.path.join(args['root'], dataset_to_str(dataset), f'{split}.jsonl')
    else:
        json_file = os.path.join(args['root'], dataset_to_str(dataset), f'{split}.json')

    assert os.path.exists(json_file), "The GT annotation file is missing!"

    annotations = {}

    if dataset == Dataset.kinetics:
        with open(json_file, 'r', encoding='utf-8') as reader:
            for line in reader:
                read_line = json.loads(line)
                key, value = list(read_line.items())[0]
                annotations[key] = value

    else:
        with open(json_file) as f:
            annotations = json.load(f)

    index = {}
    all_annotated_videos = len(annotations.keys())
    for _i, _video_id in enumerate(annotations.keys()):
        video_name = annotations[_video_id]['video_name']

        if video_name is None:
            print(f"The name of {_video_id} video is None!")
            continue

        # Debugging...

        # if _i < 2208:
        #     continue

        # if _video_id != '00707bf19d5699042dd4d79e2ce066f1':
        #     continue

        if _video_id not in index:
            index[_video_id] = f"{_i:05d}"

        frames_path = os.path.join(path, 'JPEGImages', f"{_video_id}")
        annotations_path = os.path.join(path, 'Annotations', f"{_video_id}")

        if extract_frames:  # extracting all frames of videos...

            video_input_path = os.path.join(video_path, f"{video_name}.mp4")
            assert os.path.exists(video_input_path), f"Video file does not exist in {video_input_path}"

            if not os.path.exists(frames_path):
                os.makedirs(frames_path)

            print(
                "Extracting frames {0}/{1} for the video {2}/{3}".format((_i + 1), all_annotated_videos, _video_id,
                                                                         video_name))
            extract_video_frames(video_input_path, frames_path)

        if extract_annotations:  # extracting annotations from files...

            if not os.path.exists(annotations_path):
                os.makedirs(annotations_path)

            print(
                "Extracting annotations {0}/{1} for the video {2}/{3}".format((_i + 1), all_annotated_videos,
                                                                              _video_id,
                                                                              video_name))
            point_annotations = dict()
            for point_ann in annotations[_video_id]['annotated_points']:
                frame_id = point_ann['frame_id']
                if frame_id is None:
                    continue
                if frame_id in point_annotations.keys():
                    point_annotations[frame_id].append(
                        {'x': point_ann['x'], 'y': point_ann['y'], 'status': point_ann['status'],
                         'is_fg_candidate': point_ann['is_fg_candidate']})
                else:
                    point_annotations[frame_id] = list()
                    point_annotations[frame_id].append(
                        {'x': point_ann['x'], 'y': point_ann['y'], 'status': point_ann['status'],
                         'is_fg_candidate': point_ann['is_fg_candidate']})

            extract_video_point_annotations(point_annotations, frames_path, annotations_path)

        if extract_sparse_frames:  # extracting sparse frames of videos, i.e. frames with annotations, i.e. at most 10 frames...

            sparse_frames_path = os.path.join(path, 'JPEGImages_Sparse', f"{_video_id}")
            if not os.path.exists(sparse_frames_path):
                os.makedirs(sparse_frames_path)

            print(
                "Extracting sparse frames {0}/{1} for the video {2}/{3}".format((_i + 1), all_annotated_videos,
                                                                                _video_id,
                                                                                video_name))

            extract_video_sparse_frames(frames_path, annotations_path, sparse_frames_path)

        if vis_annotations:  # visualize annotations

            visualizations_path = os.path.join(path, 'Visualizations', f"{_video_id}")
            if not os.path.exists(visualizations_path):
                os.makedirs(visualizations_path)

            print(
                "Visualizing annotations {0}/{1} for the video {2}/{3}".format((_i + 1), all_annotated_videos,
                                                                               _video_id,
                                                                               video_name))
            visualize_video_point_annotations(frames_path, annotations_path, visualizations_path)
