import functools
import logging
import multiprocessing as mp
import os

import numpy as np
import pycocotools.mask as mask_util
from PIL import Image
from detectron2.structures import BoxMode
from detectron2.utils.comm import get_world_size
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    pass

logger = logging.getLogger(__name__)


def _get_kitti360_files(kitti_root, image_gt_list_file):
    files = []
    image_dir = os.path.join(kitti_root, image_gt_list_file)
    img_gt_list = open(image_dir, "r").read().splitlines()
    # Example image_file: data_2d_raw/2013_05_28_drive_0010_sync/image_00/data_rect/0000002863.png
    # Example gt_file: data_2d_semantics/train/2013_05_28_drive_0010_sync/image_00/semantic/0000002863.png
    for img_gt in img_gt_list:
        if ' ' in img_gt:
            image_file, semantic_file = img_gt.split(' ')
            instance_file = semantic_file.replace('/semantic/', '/instance/')
            confidence_file = semantic_file.replace('/semantic/', '/confidence/')
        else:
            image_file = img_gt
            instance_file = None
            confidence_file = None
        files.append(tuple(os.path.join(kitti_root, f) for f in (image_file, instance_file, instance_file, confidence_file)))

    assert len(files), "No images found in {}".format(image_dir)

    for f in files[0]:
        if f is not None:
            assert PathManager.isfile(f), f
    return files


def load_kitti360_instances(root, image_gt_list_file, to_polygons=True):
    files = _get_kitti360_files(root, image_gt_list_file)

    logger.info("Preprocessing kitti360 annotations ...")
    # This is still not fast: all workers will execute duplicate works and will
    # take up to 10m on a 8GPU server.
    pool = mp.Pool(processes=max(mp.cpu_count() // get_world_size() // 2, 4))

    ret = pool.map(
        functools.partial(_kitti360_files_to_dict, to_polygons=to_polygons),
        files,
    )
    logger.info("Loaded {} images from {}".format(len(ret), os.path.join(root, image_gt_list_file)))

    # Map kitti360 ids to contiguous ids
    from kitti360scripts.helpers.labels import labels

    labels = [k for k in labels if k.hasInstances and not k.ignoreInInst and k.name not in ['train', 'bus', 'garage']]
    dataset_id_to_contiguous_id = {l.id: idx for idx, l in enumerate(labels)}
    for dict_per_image in ret:
        for anno in dict_per_image["annotations"]:
            # To contiguous ids, 0,1,2...
            anno["category_id"] = dataset_id_to_contiguous_id[anno["category_id"]]
    return ret


def _kitti360_files_to_dict(files, to_polygons):
    from kitti360scripts.helpers.labels import id2label, name2label

    image_file, instance_file, instance_file, confidence_file = files

    annos = []
    # See also the official annotation parsing scripts at
    # https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/evaluation/semantic_2d/instances2dict.py  # noqa

    with PathManager.open(instance_file, "rb") as f:
        inst_image = np.asarray(Image.open(f))
    # In KITTI-360, instanceId is greater than 1000. If a label does not have instances, instanceId%1000=0
    # Discard labels that do not have instancess
    flattened_ids = np.unique(inst_image[(inst_image >= 11 * 1000) & (inst_image < 42 * 1000)])

    ret = {
        "file_name": image_file,
        "image_id": os.path.basename(image_file),
        "height": inst_image.shape[0],
        "width": inst_image.shape[1],
    }

    for instance_id in flattened_ids:
        # For non-crowd annotations, instance_id // 1000 is the label_id
        # Crowd annotations have <1000 instance ids
        label_id = instance_id // 1000 if instance_id >= 1000 else instance_id
        label = id2label[label_id]
        # TODO: should we ignore 'bus' and 'train', and use label.ignoreInInst?
        # train and bus are ignored in baseline since train set doesn't have many samples of them
        # https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/evaluation/semantic_2d/evalInstanceLevelSemanticLabeling.py#L190
        if not label.hasInstances or label.ignoreInInst or label.name in ['train', 'bus']:
            continue

        anno = {}
        anno["category_id"] = label.id
        # https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/evaluation/semantic_2d/instances2dict.py
        # Merge garage instances to building instances
        # Cityscapes don't have garage label
        if anno['category_id'] == name2label['garage'].id:
            anno['category_id'] = name2label['building'].id

        mask = np.asarray(inst_image == instance_id, dtype=np.uint8, order="F")

        inds = np.nonzero(mask)
        ymin, ymax = inds[0].min(), inds[0].max()
        xmin, xmax = inds[1].min(), inds[1].max()
        anno["bbox"] = (xmin, ymin, xmax, ymax)
        if xmax <= xmin or ymax <= ymin:
            continue
        anno["bbox_mode"] = BoxMode.XYXY_ABS
        if to_polygons:
            # This conversion comes from D4809743 and D5171122,
            # when Mask-RCNN was first developed.
            contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[
                -2
            ]
            polygons = [c.reshape(-1).tolist() for c in contours if len(c) >= 3]
            # opencv's can produce invalid polygons
            if len(polygons) == 0:
                continue
            anno["segmentation"] = polygons
        else:
            anno["segmentation"] = mask_util.encode(mask[:, :, None])[0]
        annos.append(anno)

    ret["annotations"] = annos
    return ret


if __name__ == "__main__":
    """
    Test the KITTI-360 dataset loader.

    Usage:
        python -m detectron2.data.datasets.kitti360 \
            ~/KITTI-360/data_2d_semantics/train/2013_05_28_drive_val_frames.txt
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("kitti_dataset_root")
    parser.add_argument("ground_truth_list_file")
    parser.add_argument("--type", choices=["instance"], default="instance")
    args = parser.parse_args()
    from detectron2.data.catalog import Metadata
    from detectron2.utils.visualizer import Visualizer
    from kitti360scripts.helpers.labels import labels

    logger = setup_logger(name=__name__)

    dirname = "kitti360-data-vis"
    os.makedirs(dirname, exist_ok=True)

    if args.type == "instance":
        dicts = load_kitti360_instances(args.kitti_dataset_root, args.ground_truth_list_file)
        logger.info("Done loading {} samples.".format(len(dicts)))

        thing_classes = [k.name for k in labels
                         if k.hasInstances and not k.ignoreInInst and k.name not in ['train', 'bus']]
        meta = Metadata().set(thing_classes=thing_classes)

    else:
        raise NotImplementedError(f"{args.type} is not implemented yet. Try 'instance'")

    logger.info("Visualizer is being prepared...".format(len(dicts)))
    for d in dicts:
        img = np.array(Image.open(PathManager.open(d["file_name"], "rb")))
        visualizer = Visualizer(img, metadata=meta)
        vis = visualizer.draw_dataset_dict(d)
        # cv2.imshow("a", vis.get_image()[:, :, ::-1])
        # cv2.waitKey()
        fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
        vis.save(fpath)
