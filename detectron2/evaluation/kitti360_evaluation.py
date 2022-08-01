# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
import sys
import tempfile
from collections import OrderedDict
from io import StringIO

import torch
from PIL import Image

from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
from .evaluator import DatasetEvaluator


class Kitti360Evaluator(DatasetEvaluator):
    """
    Base class for evaluation using kitti360 API.
    """

    def __init__(self, dataset_name):
        """
        Args:
            dataset_name (str): the name of the dataset.
                It must have the following metadata associated with it:
                "thing_classes", "gt_dir".
        """
        self._metadata = MetadataCatalog.get(dataset_name)
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

    def reset(self):
        self._working_dir = tempfile.TemporaryDirectory(prefix="kitti360_eval_")
        self._temp_dir = self._working_dir.name
        # All workers will write to the same results directory
        # TODO this does not work in distributed training
        assert (
                comm.get_local_size() == comm.get_world_size()
        ), "Kitti360Evaluator currently do not work with multiple machines."
        self._temp_dir = comm.all_gather(self._temp_dir)[0]
        if self._temp_dir != self._working_dir.name:
            self._working_dir.cleanup()
        self._logger.info(
            "Writing kitti360 results to temporary directory {} ...".format(self._temp_dir)
        )


class Kitti360InstanceEvaluator(Kitti360Evaluator):
    """
    Evaluate instance segmentation results on kitti360 dataset using kitti360 API.

    Note:
        * It does not work in multi-machine distributed training.
        * It contains a synchronization, therefore has to be used on all ranks.
        * Only the main process runs evaluation.
    """

    def process(self, inputs, outputs):
        from kitti360scripts.helpers.labels import name2label

        for input, output in zip(inputs, outputs):
            file_name = input["file_name"]
            frameNb = os.path.splitext(os.path.basename(file_name))[0]
            sequenceNb = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(file_name)))).split('_')[-2]
            pred_txt = os.path.join(self._temp_dir, sequenceNb + "_" + frameNb + "_pred.txt")

            if "instances" in output:
                output = output["instances"].to(self._cpu_device)
                num_instances = len(output)
                with open(pred_txt, "w") as fout:
                    for i in range(num_instances):
                        pred_class = output.pred_classes[i]
                        classes = self._metadata.thing_classes[pred_class]
                        class_id = name2label[classes].id
                        score = output.scores[i]
                        mask = output.pred_masks[i].numpy().astype("uint8")
                        png_filename = os.path.join(
                            self._temp_dir, sequenceNb + "_" + frameNb + "_{}_{}.png".format(i, classes)
                        )

                        Image.fromarray(mask * 255).save(png_filename)
                        fout.write(
                            "{} {} {}\n".format(os.path.basename(png_filename), class_id, score)
                        )
            else:
                # Kitti360 requires a prediction file for every ground truth image.
                with open(pred_txt, "w") as fout:
                    pass

    def evaluate(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP" and "AP50".
        """
        comm.synchronize()
        if comm.get_rank() > 0:
            return
        import kitti360scripts.evaluation.semantic_2d.evalInstanceLevelSemanticLabeling as kitti360_eval

        self._logger.info("Evaluating results under {} ...".format(self._temp_dir))

        # set some global states in kitti360 evaluation API, before evaluating
        kitti360_eval.args.kitti360Path = os.environ[
            'KITTI360_DATASET'] if 'KITTI360_DATASET' in os.environ else self._metadata.root
        kitti360_eval.args.groundTruthListFile = os.path.join(self._metadata.root, self._metadata.gt_file)
        kitti360_eval.args.predictionPath = os.path.abspath(self._temp_dir)
        kitti360_eval.args.predictionWalk = None
        kitti360_eval.args.JSONOutput = False
        kitti360_eval.args.colorized = False
        kitti360_eval.args.quiet = False
        kitti360_eval.args.gtInstancesFile = os.path.join(self._temp_dir, "gtInstances.json")

        # These lines are adopted from
        # https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/evaluation/semantic_2d/evalInstanceLevelSemanticLabeling.py # noqa
        with PathManager.open(kitti360_eval.args.groundTruthListFile, "r") as f:
            pairs = f.read().splitlines()

        groundTruthImgList = []
        for i, pair in enumerate(pairs):
            groundTruthFile = os.path.join(kitti360_eval.args.kitti360Path, pair.split(' ')[1])

            groundTruthFile = os.path.join(os.path.dirname(os.path.dirname(groundTruthFile)),
                                           'instance', os.path.basename(groundTruthFile))
            assert PathManager.isfile(groundTruthFile), \
                f"Could not open '{groundTruthFile}'. Please read the instructions of this method."

            confidenceFile = os.path.join(os.path.dirname(os.path.dirname(groundTruthFile)),
                                          'confidence', os.path.basename(groundTruthFile))
            assert PathManager.isfile(confidenceFile), \
                f"Could not open '{confidenceFile}'. Please download the confidence maps for evaluation."

            groundTruthImgList.append([groundTruthFile, confidenceFile])
        assert len(
            groundTruthImgList
        ), "Cannot find any ground truth images to use for evaluation. Searched for: {}".format(
            kitti360_eval.args.groundTruthListFile
        )

        predictionImgList = []
        for gt, _ in groundTruthImgList:
            predictionImgList.append(kitti360_eval.getPrediction(kitti360_eval.args, gt))
        results = kitti360_eval.evaluateImgLists(
            predictionImgList, groundTruthImgList, kitti360_eval.args
        )['averages']

        # Log printed results
        old_stdout = sys.stdout
        result = StringIO()
        sys.stdout = result
        kitti360_eval.printResults(results, kitti360_eval.args)
        self._logger.info(result.getvalue())
        sys.stdout = old_stdout

        ret = OrderedDict()
        ret["segm"] = {
            "allAP": results["allAp"] * 100,
            "allAP50": results["allAp50%"] * 100,
            "carAP": results["classes"]["car"]["ap"] * 100,
            "carAP50": results["classes"]["car"]["ap50%"] * 100
        }
        self._working_dir.cleanup()
        return ret
