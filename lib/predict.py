import os
import torch
import numpy as np

from PIL import Image

from .utils import idx2onehot


def predict(ref,
            target,
            ref_label,
            weight_dense,
            weight_sparse,
            frame_idx,
            args):
    """
    The Predict Function.
    :param ref: (N, feature_dim, H, W)
    :param target: (feature_dim, H, W)
    :param ref_label: (d, N, H*W)
    :param weight_dense: (H*W, H*W)
    :param weight_sparse: (H*W, H*W)
    :param frame_idx:
    :param args:
    :return: (d, H, W)
    """
    # sample frames from history features
    d = ref_label.shape[0]
    sample_idx = sample_frames(frame_idx, args.range, args.ref_num)
    ref_selected = ref.index_select(0, sample_idx)  # (n, C, H/8, W/8)
    ref_label_selected = ref_label.index_select(1, sample_idx).view(d, -1)  # (d, n, H/8*W/8) --> (d, n*H/8*W/8)

    # get similarity matrix
    (num_ref, feature_dim, H, W) = ref_selected.shape
    ref_selected = ref_selected.permute(0, 2, 3, 1).reshape(-1, feature_dim)  # (n*H/8*W/8, C)
    target = target.reshape(feature_dim, -1)  # (C, H/8*W/8)
    global_similarity = ref_selected.mm(target)  # (n*H/8*W/8, H/8*W/8)

    # temperature step
    global_similarity *= args.temperature

    # softmax
    # TODO: 这里应该是先计算相似度，可以做exp激活，但是不应该直接softmax。应该把motion的影响作用到exp激活后的结果上，而不是作用在概率上
    global_similarity = global_similarity.softmax(dim=0)

    # spatial weight and motion model
    global_similarity = global_similarity.contiguous().view(num_ref, H * W, H * W)  # (n, H/8*W/8, H/8*W/8)
    # 这里还有一处细节，前15帧只用short-term memory, 15帧以后才开始同时使用short-term和long-term memory
    if frame_idx > 15:
        continuous_frame = 4
        # interval frames
        global_similarity[:-continuous_frame] *= weight_sparse
        # continuous frames
        global_similarity[-continuous_frame:] *= weight_dense
    else:
        global_similarity = global_similarity.mul(weight_dense)
    global_similarity = global_similarity.view(-1, H * W)  # (n*H/8*W/8, H/8*W/8)
    # TODO: 这里应该重新归一化一次，不然其实就不满足转移矩阵的要求了
    # get prediction
    prediction = ref_label_selected.mm(global_similarity)  # (d, n*H/8*W/8) * (n*H/8*W/8, H/8*W/8) --> (d, H/8*W/8)
    return prediction


def sample_frames(frame_idx,
                  take_range,
                  num_refs):
    # sample dense and sparse reference frames
    if frame_idx <= num_refs:
        sample_idx = list(range(frame_idx))
    else:
        dense_num = 4 - 1
        sparse_num = num_refs - dense_num
        target_idx = frame_idx
        ref_end = target_idx - dense_num - 1
        ref_start = max(ref_end - take_range, 0)
        sample_idx = np.linspace(ref_start, ref_end, sparse_num).astype(np.int).tolist()
        for j in range(dense_num):
            sample_idx.append(target_idx - dense_num + j)

    return torch.Tensor(sample_idx).long().cuda()


def prepare_first_frame(curr_video,
                        save_prediction,
                        annotation_dir,
                        sigma1=8,
                        sigma2=21):
    annotation_list = sorted(os.listdir(annotation_dir))
    first_annotation = Image.open(os.path.join(annotation_dir, annotation_list[curr_video], '00000.png'))
    (H, W) = np.asarray(first_annotation).shape
    H_d = int(np.ceil(H / 8))  # stride=8, use ceil method
    W_d = int(np.ceil(W / 8))
    palette = first_annotation.getpalette()
    label = np.asarray(first_annotation)  # for the sequence bike-packing, label only has three types of values: 0, 1, 2
    d = np.max(label) + 1
    label = torch.Tensor(label).long().cuda()  # (1, H, W)
    label_1hot = idx2onehot(label.view(-1), d).reshape(1, d, H, W)  # (1, d, H, W)
    # after the interpolation, values of down-sampled label become real numbers rather than int numbers.
    # but it is to prove that the down-sampled label is still a probability distribution
    label_1hot = torch.nn.functional.interpolate(label_1hot,
                                                           size=(H_d, W_d),
                                                           mode='bilinear',
                                                           align_corners=False)
    label_1hot = label_1hot.reshape(d, -1).unsqueeze(1)  # (d, 1, HW)
    weight_dense = get_spatial_weight((H_d, W_d), sigma1)
    weight_sparse = get_spatial_weight((H_d, W_d), sigma2)

    if save_prediction is not None:
        if not os.path.exists(save_prediction):
            os.makedirs(save_prediction)
        save_path = os.path.join(save_prediction, annotation_list[curr_video])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        first_annotation.save(os.path.join(save_path, '00000.png'))

    return label_1hot, d, palette, weight_dense, weight_sparse


def get_spatial_weight(shape, sigma):
    """
    Get soft spatial weights for similarity matrix.
    :param shape: (H, W)
    :param sigma:
    :return: (H*W, H*W)
    """
    (H, W) = shape

    index_matrix = torch.arange(H * W, dtype=torch.long).reshape(H * W, 1).cuda()
    index_matrix = torch.cat((index_matrix / W, index_matrix % W), -1)  # (H*W, 2)
    d = index_matrix - index_matrix.unsqueeze(1)  # (H*W, H*W, 2)
    d = d.float().pow(2).sum(-1)  # (H*W, H*W)
    w = (- d / sigma ** 2).exp()

    return w
