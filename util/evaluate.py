import torch
import numpy as np
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log


def evaluate(model, loader, mode, cfg):
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    with torch.no_grad():
        for img, dem, mask, id in loader:

            img = img.cuda()
            dem = dem.cuda()

            grid = cfg['crop_size']
            b, _, h, w = img.shape
            final = torch.zeros(b, cfg['nclass'], h, w).cuda()

            if dem.shape[1] != 1 or img.shape[1] != 3:
                img1, img2 = torch.chunk(img, 2, dim=1)
                dem1, dem2 = torch.chunk(dem, 2, dim=1)
                img1, img2 = img1.cuda(), img2.cuda()
                dem1, dem2 = dem1.cuda(), dem2.cuda()

                pred1 = model(img1, dem1)
                pred2 = model(img1, dem2)
                pred3 = model(img2, dem1)
                pred4 = model(img2, dem2)
                final = (pred1.softmax(dim=1) + pred2.softmax(dim=1)+pred3.softmax(dim=1)+pred4.softmax(dim=1)) / 4

            else:
                pred = model(img, dem)
                final = pred.softmax(dim=1)


            pred = final.argmax(dim=1)


            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()
            reduced_target = torch.from_numpy(target).cuda()

            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIOU = np.mean(iou_class)

    return mIOU, iou_class


