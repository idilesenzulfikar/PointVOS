"""
This script can be get the statistics for the given dataset.
Supported datasets are Kinetics, Oops
"""
import glob
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from PIL import Image

from dataset_scripts.keys import Dataset, dataset_to_str, str_to_dataset


BACKGROUND_CLASS_ID = 0
REJECTED_CLASS_ID = 1
ACCEPTED_CLASS_ID = 2
AMBIGUOUS_CLASS_ID = 255

if __name__ == '__main__':
    parser = ArgumentParser()

    # Dataset Parameters
    parser.add_argument('--root', required=True, type=str, help='The root directory for the given dataset.')
    parser.add_argument('--dataset', required=True, type=str, choices=[e.value for e in Dataset],
                        help='The name of the given dataset.')
    parser.add_argument('--split', required=True, type=str, choices=['train', 'val'], help='The dataset split.')

    # Output Parameters
    parser.add_argument('--output_root', required=True, type=str,
                        help="The output root to save excel files for statistics.")

    # Annotation Parameters
    parser.add_argument('--rejected_class_id',
                        default=REJECTED_CLASS_ID, type=int,
                        help='Rejected points class id that represents background points in the dataset')
    parser.add_argument('--accepted_class_id', default=ACCEPTED_CLASS_ID, type=int,
                        help='Accepted points class id that represents foreground points in the dataset')
    parser.add_argument('--ambiguous_class_id', default=AMBIGUOUS_CLASS_ID, type=int,
                        help='Ambigous points class id that represents not annotated points in the dataset')

    # Statistics Parameters
    # Total
    parser.add_argument('--total_video', action='store_true',
                        help='Compute the total number of videos/frames/annotations/(rejected,accepted,ambiguous) points in the given split of the dataset')
    # Per-video
    parser.add_argument('--per_video', action='store_true',
                        help='Compute the number of frames/annotations/(rejected,accepted,ambiguous) points for each video in the given split of the dataset')

    args = vars(parser.parse_args())

    # Get Dataset Parameters
    assert os.path.exists(args['root']), "The dataset root does not exist!!"

    dataset = str_to_dataset(args['dataset'])
    split = args['split']

    # dataset_annotations = os.path.join(args['root'], annotation_supervision_to_str(annotation_supervision),
    #                                    'runs', dataset_to_str(dataset), split, 'Annotations')
    # dataset_frames = os.path.join(args['root'], annotation_supervision_to_str(annotation_supervision), 'runs',
    #                               dataset_to_str(dataset), split, 'JPEGImages')

    dataset_annotations = os.path.join(args['root'], dataset_to_str(dataset), split, 'Annotations')
    dataset_frames = os.path.join(args['root'], dataset_to_str(dataset), split, 'JPEGImages')

    # Get output parameters
    output_root = args['output_root']
    assert os.path.exists(output_root), 'The output root does not exist!!'

    print("Extracting the statistics for the given dataset:")

    # Get statistics
    all_videos = glob.glob(dataset_frames + '/*')

    total_frames, total_annotations = 0, 0
    total_rejected_points, total_accepted_points, total_ambiguous_points = 0, 0, 0

    per_video_statistics = dict()

    for video_path in all_videos:
        video = video_path.split('/')[-1]

        video_frames = glob.glob(dataset_frames + '/' + video + '/*.jpg')
        video_annotations = glob.glob(dataset_annotations + '/' + video + '/*.png')

        total_frames = total_frames + len(video_frames)
        total_annotations = total_annotations + len(video_annotations)

        per_video_statistics[video] = dict()
        per_video_statistics[video]['total_frames'] = len(video_frames)
        per_video_statistics[video]['total_annotations'] = len(video_annotations)

        vid_rejected_points, vid_accepted_points, vid_ambiguous_points = 0, 0, 0
        vid_frames_w_accepted_points = 0
        for ann in video_annotations:
            ann_mask = np.array(Image.open(ann).convert('P'), dtype=np.uint8)
            ann_range, ann_count = np.unique(ann_mask, return_counts=True)

            rejected_points = ann_count[np.where(ann_range == REJECTED_CLASS_ID)[0]][0] if len(
                ann_count[np.where(ann_range == REJECTED_CLASS_ID)]) > 0 else 0
            total_rejected_points = total_rejected_points + rejected_points
            vid_rejected_points = vid_rejected_points + rejected_points

            accepted_points = ann_count[np.where(ann_range == ACCEPTED_CLASS_ID)[0]][0] if len(
                ann_count[np.where(ann_range == ACCEPTED_CLASS_ID)]) > 0 else 0
            total_accepted_points = total_accepted_points + accepted_points
            vid_accepted_points = vid_accepted_points + accepted_points
            if accepted_points > 0: vid_frames_w_accepted_points += 1

            ambiguous_points = ann_count[np.where(ann_range == AMBIGUOUS_CLASS_ID)[0]][0] if len(
                ann_count[np.where(ann_range == AMBIGUOUS_CLASS_ID)]) > 0 else 0
            total_ambiguous_points = total_ambiguous_points + ambiguous_points
            vid_ambiguous_points = vid_ambiguous_points + ambiguous_points

        per_video_statistics[video]['total_rejected_points'] = vid_rejected_points
        per_video_statistics[video]['total_accepted_points'] = vid_accepted_points
        per_video_statistics[video]['total_ambiguous_points'] = vid_ambiguous_points
        per_video_statistics[video]['frames_w_accepted_points'] = vid_frames_w_accepted_points

    # Print Total Statistics
    if args['total_video']:
        print('-' * 80)
        print('*' * 5 + ' Extracting statistics for the {0} split of {1} dataset '.format(split, dataset_to_str(
            dataset)) + '*' * 5)
        print('-' * 80)

        print('Total number of videos: {0}'.format(len(all_videos)))
        print('Total number of frames: {0}'.format(total_frames))
        print('Total number of annotated frames: {0}'.format(total_annotations))
        print('Total number of rejected points: {0}'.format(total_rejected_points))
        print('Total number of accepted points: {0}'.format(total_accepted_points))
        print('Total number of ambiguous points: {0}'.format(total_ambiguous_points))

        print('-' * 80)

    # Print Per-video Statistics
    if args['per_video']:
        print('*' * 5 + ' Saving per video statistics into excel for the {0} split of {1} dataset '.format(split,
                                                                                                           dataset_to_str(
                                                                                                               dataset)) + '*' * 5)

        per_video_total_frames = list()
        per_video_annoted_frames = list()
        per_video_frames_with_accepted_points = list()
        per_video_points = list()

        per_rejected_points = dict()
        per_accepted_points = dict()
        per_ambiguous_points = dict()

        per_frame_with_accepted_points = {'0-frame': 0, '1-frame': 0, '2-frame': 0, '3-frame': 0, '4-frame': 0,
                                          '5-frame': 0, '6-frame': 0,
                                          '7-frame': 0, '8-frame': 0, '9-frame': 0, '10-frame': 0}

        for video, statistics in per_video_statistics.items():
            per_video_total_frames.append(statistics['total_frames'])
            per_video_annoted_frames.append(statistics['total_annotations'])
            per_video_frames_with_accepted_points.append(statistics['frames_w_accepted_points'])
            per_video_points.append([statistics['total_rejected_points'], statistics['total_accepted_points'],
                                     statistics['total_ambiguous_points']])
            if statistics['total_rejected_points'] not in per_rejected_points.keys():
                per_rejected_points[statistics['total_rejected_points']] = 1
            else:
                per_rejected_points[statistics['total_rejected_points']] += 1

            if statistics['total_accepted_points'] not in per_accepted_points.keys():
                per_accepted_points[statistics['total_accepted_points']] = 1
            else:
                per_accepted_points[statistics['total_accepted_points']] += 1

            if statistics['total_ambiguous_points'] not in per_ambiguous_points.keys():
                per_ambiguous_points[statistics['total_ambiguous_points']] = 1
            else:
                per_ambiguous_points[statistics['total_ambiguous_points']] += 1

            if statistics['frames_w_accepted_points'] == 0:
                per_frame_with_accepted_points['0-frame'] += 1
            elif statistics['frames_w_accepted_points'] == 1:
                per_frame_with_accepted_points['1-frame'] += 1
            elif statistics['frames_w_accepted_points'] == 2:
                per_frame_with_accepted_points['2-frame'] += 1
            elif statistics['frames_w_accepted_points'] == 3:
                per_frame_with_accepted_points['3-frame'] += 1
            elif statistics['frames_w_accepted_points'] == 4:
                per_frame_with_accepted_points['4-frame'] += 1
            elif statistics['frames_w_accepted_points'] == 5:
                per_frame_with_accepted_points['5-frame'] += 1
            elif statistics['frames_w_accepted_points'] == 6:
                per_frame_with_accepted_points['6-frame'] += 1
            elif statistics['frames_w_accepted_points'] == 7:
                per_frame_with_accepted_points['7-frame'] += 1
            elif statistics['frames_w_accepted_points'] == 8:
                per_frame_with_accepted_points['8-frame'] += 1
            elif statistics['frames_w_accepted_points'] == 9:
                per_frame_with_accepted_points['9-frame'] += 1
            elif statistics['frames_w_accepted_points'] == 10:
                per_frame_with_accepted_points['10-frame'] += 1

        # sort per_point analysis
        per_rejected_points = dict(sorted(per_rejected_points.items()))
        per_accepted_points = dict(sorted(per_accepted_points.items()))
        per_ambiguous_points = dict(sorted(per_ambiguous_points.items()))

        rejected_points_in_range = {'0-10': 0, '10-20': 0, '20-30': 0, '30-40': 0, '40-50': 0, '50-60': 0, '60-70': 0,
                                    '70-80': 0, '80-90': 0, '90-100': 0, '100-110': 0, '110-120': 0, '120-130': 0,
                                    '130-140': 0, '140-150': 0, '150-160': 0, '160-170': 0, '170-180': 0, '180-190': 0,
                                    '190-200': 0}

        for rejected_points, total_video in per_rejected_points.items():
            if 0 <= rejected_points <= 10:
                rejected_points_in_range['0-10'] = rejected_points_in_range['0-10'] + total_video
            elif 10 < rejected_points <= 20:
                rejected_points_in_range['10-20'] = rejected_points_in_range['10-20'] + total_video
            elif 20 < rejected_points <= 30:
                rejected_points_in_range['20-30'] = rejected_points_in_range['20-30'] + total_video
            elif 30 < rejected_points <= 40:
                rejected_points_in_range['30-40'] = rejected_points_in_range['30-40'] + total_video
            elif 40 < rejected_points <= 50:
                rejected_points_in_range['40-50'] = rejected_points_in_range['40-50'] + total_video
            elif 50 < rejected_points <= 60:
                rejected_points_in_range['50-60'] = rejected_points_in_range['50-60'] + total_video
            elif 60 < rejected_points <= 70:
                rejected_points_in_range['60-70'] = rejected_points_in_range['60-70'] + total_video
            elif 70 < rejected_points <= 80:
                rejected_points_in_range['70-80'] = rejected_points_in_range['70-80'] + total_video
            elif 80 < rejected_points <= 90:
                rejected_points_in_range['80-90'] = rejected_points_in_range['80-90'] + total_video
            elif 90 < rejected_points <= 100:
                rejected_points_in_range['90-100'] = rejected_points_in_range['90-100'] + total_video
            elif 100 < rejected_points <= 110:
                rejected_points_in_range['100-110'] = rejected_points_in_range['100-110'] + total_video
            elif 110 < rejected_points <= 120:
                rejected_points_in_range['110-120'] = rejected_points_in_range['110-120'] + total_video
            elif 120 < rejected_points <= 130:
                rejected_points_in_range['120-130'] = rejected_points_in_range['120-130'] + total_video
            elif 130 < rejected_points <= 140:
                rejected_points_in_range['130-140'] = rejected_points_in_range['130-140'] + total_video
            elif 140 < rejected_points <= 150:
                rejected_points_in_range['140-150'] = rejected_points_in_range['140-150'] + total_video
            elif 150 < rejected_points <= 160:
                rejected_points_in_range['150-160'] = rejected_points_in_range['150-160'] + total_video
            elif 160 < rejected_points <= 170:
                rejected_points_in_range['160-170'] = rejected_points_in_range['160-170'] + total_video
            elif 170 < rejected_points <= 180:
                rejected_points_in_range['170-180'] = rejected_points_in_range['170-180'] + total_video
            elif 180 < rejected_points <= 190:
                rejected_points_in_range['180-190'] = rejected_points_in_range['180-190'] + total_video
            elif 190 < rejected_points <= 200:
                rejected_points_in_range['190-200'] = rejected_points_in_range['190-200'] + total_video

        accepted_points_in_range = {'0-10': 0, '10-20': 0, '20-30': 0, '30-40': 0, '40-50': 0, '50-60': 0, '60-70': 0,
                                    '70-80': 0, '80-90': 0, '90-100': 0, '100-110': 0, '110-120': 0, '120-130': 0,
                                    '130-140': 0, '140-150': 0, '150-160': 0, '160-170': 0, '170-180': 0, '180-190': 0,
                                    '190-200': 0}

        for accepted_points, total_video in per_accepted_points.items():
            if 0 <= accepted_points <= 10:
                accepted_points_in_range['0-10'] = accepted_points_in_range['0-10'] + total_video
            elif 10 < accepted_points <= 20:
                accepted_points_in_range['10-20'] = accepted_points_in_range['10-20'] + total_video
            elif 20 < accepted_points <= 30:
                accepted_points_in_range['20-30'] = accepted_points_in_range['20-30'] + total_video
            elif 30 < accepted_points <= 40:
                accepted_points_in_range['30-40'] = accepted_points_in_range['30-40'] + total_video
            elif 40 < accepted_points <= 50:
                accepted_points_in_range['40-50'] = accepted_points_in_range['40-50'] + total_video
            elif 50 < accepted_points <= 60:
                accepted_points_in_range['50-60'] = accepted_points_in_range['50-60'] + total_video
            elif 60 < accepted_points <= 70:
                accepted_points_in_range['60-70'] = accepted_points_in_range['60-70'] + total_video
            elif 70 < accepted_points <= 80:
                accepted_points_in_range['70-80'] = accepted_points_in_range['70-80'] + total_video
            elif 80 < accepted_points <= 90:
                accepted_points_in_range['80-90'] = accepted_points_in_range['80-90'] + total_video
            elif 90 < accepted_points <= 100:
                accepted_points_in_range['90-100'] = accepted_points_in_range['90-100'] + total_video
            elif 100 < accepted_points <= 110:
                accepted_points_in_range['100-110'] = accepted_points_in_range['100-110'] + total_video
            elif 110 < accepted_points <= 120:
                accepted_points_in_range['110-120'] = accepted_points_in_range['110-120'] + total_video
            elif 120 < accepted_points <= 130:
                accepted_points_in_range['120-130'] = accepted_points_in_range['120-130'] + total_video
            elif 130 < accepted_points <= 140:
                accepted_points_in_range['130-140'] = accepted_points_in_range['130-140'] + total_video
            elif 140 < accepted_points <= 150:
                accepted_points_in_range['140-150'] = accepted_points_in_range['140-150'] + total_video
            elif 150 < accepted_points <= 160:
                accepted_points_in_range['150-160'] = accepted_points_in_range['150-160'] + total_video
            elif 160 < accepted_points <= 170:
                accepted_points_in_range['160-170'] = accepted_points_in_range['160-170'] + total_video
            elif 170 < accepted_points <= 180:
                accepted_points_in_range['170-180'] = accepted_points_in_range['170-180'] + total_video
            elif 180 < accepted_points <= 190:
                accepted_points_in_range['180-190'] = accepted_points_in_range['180-190'] + total_video
            elif 190 < accepted_points <= 200:
                accepted_points_in_range['190-200'] = accepted_points_in_range['190-200'] + total_video

        ambiguous_points_in_range = {'0-10': 0, '10-20': 0, '20-30': 0, '30-40': 0, '40-50': 0, '50-60': 0, '60-70': 0,
                                     '70-80': 0, '80-90': 0, '90-100': 0, '100-110': 0, '110-120': 0, '120-130': 0,
                                     '130-140': 0, '140-150': 0, '150-160': 0, '160-170': 0, '170-180': 0, '180-190': 0,
                                     '190-200': 0}

        for ambiguous_points, total_video in per_ambiguous_points.items():
            if 0 <= ambiguous_points <= 10:
                ambiguous_points_in_range['0-10'] = ambiguous_points_in_range['0-10'] + total_video
            elif 10 < ambiguous_points <= 20:
                ambiguous_points_in_range['10-20'] = ambiguous_points_in_range['10-20'] + total_video
            elif 20 < ambiguous_points <= 30:
                ambiguous_points_in_range['20-30'] = ambiguous_points_in_range['20-30'] + total_video
            elif 30 < ambiguous_points <= 40:
                ambiguous_points_in_range['30-40'] = ambiguous_points_in_range['30-40'] + total_video
            elif 40 < ambiguous_points <= 50:
                ambiguous_points_in_range['40-50'] = ambiguous_points_in_range['40-50'] + total_video
            elif 50 < ambiguous_points <= 60:
                ambiguous_points_in_range['50-60'] = ambiguous_points_in_range['50-60'] + total_video
            elif 60 < ambiguous_points <= 70:
                ambiguous_points_in_range['60-70'] = ambiguous_points_in_range['60-70'] + total_video
            elif 70 < ambiguous_points <= 80:
                ambiguous_points_in_range['70-80'] = ambiguous_points_in_range['70-80'] + total_video
            elif 80 < ambiguous_points <= 90:
                ambiguous_points_in_range['80-90'] = ambiguous_points_in_range['80-90'] + total_video
            elif 90 < ambiguous_points <= 100:
                ambiguous_points_in_range['90-100'] = ambiguous_points_in_range['90-100'] + total_video
            elif 100 < ambiguous_points <= 110:
                ambiguous_points_in_range['100-110'] = ambiguous_points_in_range['100-110'] + total_video
            elif 110 < ambiguous_points <= 120:
                ambiguous_points_in_range['110-120'] = ambiguous_points_in_range['110-120'] + total_video
            elif 120 < ambiguous_points <= 130:
                ambiguous_points_in_range['120-130'] = ambiguous_points_in_range['120-130'] + total_video
            elif 130 < ambiguous_points <= 140:
                ambiguous_points_in_range['130-140'] = ambiguous_points_in_range['130-140'] + total_video
            elif 140 < ambiguous_points <= 150:
                ambiguous_points_in_range['140-150'] = ambiguous_points_in_range['140-150'] + total_video
            elif 150 < ambiguous_points <= 160:
                ambiguous_points_in_range['150-160'] = ambiguous_points_in_range['150-160'] + total_video
            elif 160 < ambiguous_points <= 170:
                ambiguous_points_in_range['160-170'] = ambiguous_points_in_range['160-170'] + total_video
            elif 170 < ambiguous_points <= 180:
                ambiguous_points_in_range['170-180'] = ambiguous_points_in_range['170-180'] + total_video
            elif 180 < ambiguous_points <= 190:
                ambiguous_points_in_range['180-190'] = ambiguous_points_in_range['180-190'] + total_video
            elif 190 < ambiguous_points <= 200:
                ambiguous_points_in_range['190-200'] = ambiguous_points_in_range['190-200'] + total_video

        ################################################################################################################

        df1 = pd.DataFrame(per_video_total_frames, index=per_video_statistics.keys(),
                           columns=['#_of_frames'])

        df2 = pd.DataFrame(per_video_annoted_frames, index=per_video_statistics.keys(),
                           columns=['#_of_annotations'])

        df3 = pd.DataFrame(per_video_frames_with_accepted_points, index=per_video_statistics.keys(),
                           columns=['#_of_frames'])

        df4 = pd.DataFrame(per_video_points,
                           index=per_video_statistics.keys(),
                           columns=['#_of_rejected_points', '#_of_accepted_points', '#_of_ambiguous_points'])

        ################################################################################################################

        df5 = pd.DataFrame(list(per_rejected_points.values()), index=per_rejected_points.keys(),
                           columns=['total_video'])

        df6 = pd.DataFrame(list(per_accepted_points.values()), index=per_accepted_points.keys(),
                           columns=['total_video'])

        df7 = pd.DataFrame(list(per_ambiguous_points.values()), index=per_ambiguous_points.keys(),
                           columns=['total_video'])

        ################################################################################################################

        df8 = pd.DataFrame(list(rejected_points_in_range.values()), index=rejected_points_in_range.keys(),
                           columns=['total_video'])

        df9 = pd.DataFrame(list(accepted_points_in_range.values()), index=accepted_points_in_range.keys(),
                           columns=['total_video'])

        df10 = pd.DataFrame(list(ambiguous_points_in_range.values()), index=ambiguous_points_in_range.keys(),
                            columns=['total_video'])

        ################################################################################################################

        df11 = pd.DataFrame(list(per_frame_with_accepted_points.values()),
                            index=per_frame_with_accepted_points.keys(), columns=['total_video'])

        # output_path = os.path.join(output_root, annotation_supervision_to_str(annotation_supervision),
        #                            'runs', dataset_to_str(dataset), split, 'Statistics')

        output_path = os.path.join(output_root, dataset_to_str(dataset), split, 'Statistics')

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        output_file = os.path.join(output_path, 'per_video_statistics.xlsx')
        with pd.ExcelWriter(output_file) as writer:
            df1.to_excel(writer, sheet_name='All Frames')
            df2.to_excel(writer, sheet_name='Annotated Frames')
            df3.to_excel(writer, sheet_name='Frames with Accepted Points')
            df4.to_excel(writer, sheet_name='Video Points')

            df5.to_excel(writer, sheet_name='Rejected Points')
            df6.to_excel(writer, sheet_name='Accepted Points')
            df7.to_excel(writer, sheet_name='Ambiguous Points')

            df8.to_excel(writer, sheet_name='Rejected Points in Range')
            df9.to_excel(writer, sheet_name='Accepted Points in Range')
            df10.to_excel(writer, sheet_name='Ambiguous Points in Range')
            df11.to_excel(writer, sheet_name='Frames with Accepted Points in Range')
        print('-' * 80)
