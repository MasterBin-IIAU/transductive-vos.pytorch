import argparse
import os
import time

import torch
import torch.nn as nn
import numpy as np

import dataset
import modeling

from lib.utils import AverageMeter, save_prediction, idx2onehot
from lib.predict import predict, prepare_first_frame
from lib.predict import predict_topk

parser = argparse.ArgumentParser()
parser.add_argument('--ref_num', '-n', type=int, default=9,
                    help='number of reference frames for inference')
parser.add_argument('--dataset', '-ds', type=str, default='davis',
                    help='name of dataset')
parser.add_argument('--data', type=str,
                    help='path to inference dataset')
parser.add_argument('--resume', '-r', type=str,
                    help='path to the resumed checkpoint')
parser.add_argument('--model', type=str, default='resnet50',
                    help='network architecture, resnet18, resnet50 or resnet101')
parser.add_argument('--temperature', '-t', type=float, default=1.0,
                    help='temperature parameter')
parser.add_argument('--range', type=int, default=40,
                    help='range of frames for inference')
parser.add_argument('--sigma1', type=float, default=8.0,
                    help='smaller sigma in the motion model for dense spatial weight')
parser.add_argument('--sigma2', type=float, default=21.0,
                    help='bigger sigma in the motion model for sparse spatial weight')
parser.add_argument('--save', '-s', type=str,
                    help='path to save predictions')
'''add an extra initial frame'''
parser.add_argument('--add_init', type=int, default=0, help="whether to add an extra initial frame")
'''use new similarity formula'''
parser.add_argument('--new_similar', type=int, default=0, help="whether to use new similarity formula")
'''only use top-k'''
parser.add_argument('--topk', type=int, default=-1, help="k nearest neighbours considered in the similarity computation")
'''add a new argument dense_num'''
parser.add_argument('--dense_num', type=int, default=4,
                    help='number of dense frames for inference')  # by default, it is 4 in the original implementation

device = torch.device("cuda")


def main():
    global args
    args = parser.parse_args()

    model = modeling.VOSNet(model=args.model)
    model = nn.DataParallel(model)
    model.cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'"
                  .format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    model.eval()

    data_dir = os.path.join(args.data, 'DAVIS_val/JPEGImages/480p')
    inference_dataset = dataset.DavisInference(data_dir)
    inference_loader = torch.utils.data.DataLoader(inference_dataset,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   num_workers=8)
    inference(inference_loader, model, args)


def inference(inference_loader, model, args):
    global pred_visualize, palette, d, feats_history, label_history, weight_dense, weight_sparse
    batch_time = AverageMeter()
    annotation_dir = os.path.join(args.data, 'DAVIS_val/Annotations/480p')
    annotation_list = sorted(os.listdir(annotation_dir))

    last_video = 0
    frame_idx = 0
    with torch.no_grad():
        for i, (input, curr_video, img_original) in enumerate(inference_loader):
            if curr_video != last_video:
                # save prediction
                pred_visualize = pred_visualize.cpu().numpy()
                for f in range(1, frame_idx):
                    save_path = args.save
                    save_name = str(f).zfill(5)
                    video_name = annotation_list[last_video]
                    save_prediction(np.asarray(pred_visualize[f - 1], dtype=np.int32),
                                    palette, save_path, save_name, video_name)

                frame_idx = 0
                print("End of video %d. Processing a new annotation..." % (last_video + 1))
            if frame_idx == 0:
                input = input.to(device)
                with torch.no_grad():
                    feats_history = model(input)
                label_history, d, palette, weight_dense, weight_sparse = prepare_first_frame(curr_video,
                                                                                                 args.save,
                                                                                                 annotation_dir,
                                                                                                 args.sigma1,
                                                                                                 args.sigma2)
                frame_idx += 1
                last_video = curr_video
                continue
            (batch_size, num_channels, H, W) = input.shape
            input = input.to(device)

            start = time.time()
            features = model(input)
            (_, feature_dim, H_d, W_d) = features.shape
            if args.topk < 0:
                prediction = predict(feats_history,
                                     features[0],
                                     label_history,
                                     weight_dense,
                                     weight_sparse,
                                     frame_idx,
                                     args
                                     )
            else:
                prediction = predict_topk(feats_history,
                                     features[0],
                                     label_history,
                                     weight_dense,
                                     weight_sparse,
                                     frame_idx,
                                     args
                                     )
            # Store all frames' features
            new_label = idx2onehot(torch.argmax(prediction, 0), d).unsqueeze(1)  # (d, H/8*W/8) --> (d, 1, H/8*W/8)
            label_history = torch.cat((label_history, new_label), 1)  # (d, L, HW) d is number of instances, L is number of frames
            feats_history = torch.cat((feats_history, features), 0)

            last_video = curr_video
            frame_idx += 1

            # 1. upsample, 2. argmax
            prediction = torch.nn.functional.interpolate(prediction.view(1, d, H_d, W_d),
                                                         size=(H, W),
                                                         mode='bilinear',
                                                         align_corners=False)
            prediction = torch.argmax(prediction, 1)  # (1, H, W)

            if frame_idx == 2:
                pred_visualize = prediction
            else:
                pred_visualize = torch.cat((pred_visualize, prediction), 0)  # (L, H, W)

            batch_time.update(time.time() - start)

            if i % 10 == 0:
                print('Validate: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                    i, len(inference_loader), batch_time=batch_time))
        # save last video's prediction
        pred_visualize = pred_visualize.cpu().numpy()
        for f in range(1, frame_idx):
            save_path = args.save
            save_name = str(f).zfill(5)
            video_name = annotation_list[last_video]
            save_prediction(np.asarray(pred_visualize[f - 1], dtype=np.int32),
                            palette, save_path, save_name, video_name)
    print('Finished inference.')


if __name__ == '__main__':
    main()
