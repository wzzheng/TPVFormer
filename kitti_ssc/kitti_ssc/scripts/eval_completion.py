#!/usr/bin/python3

import argparse
import numpy as np
import scipy.io as sio
import yaml
import os
import time

epsilon = np.finfo(np.float32).eps


def get_eval_mask(labels, invalid_voxels):
  """
  Ignore labels set to 255 and invalid voxels (the ones never hit by a laser ray, probed using ray tracing)
  :param labels: input ground truth voxels
  :param invalid_voxels: voxels ignored during evaluation since the lie beyond the scene that was captured by the laser
  :return: boolean mask to subsample the voxels to evaluate
  """
  masks = np.ones_like(labels, dtype=np.bool)
  masks[labels == 255] = False
  masks[invalid_voxels == 1] = False

  return masks


def unpack(compressed):
  ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
  uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
  uncompressed[::8] = compressed[:] >> 7 & 1
  uncompressed[1::8] = compressed[:] >> 6 & 1
  uncompressed[2::8] = compressed[:] >> 5 & 1
  uncompressed[3::8] = compressed[:] >> 4 & 1
  uncompressed[4::8] = compressed[:] >> 3 & 1
  uncompressed[5::8] = compressed[:] >> 2 & 1
  uncompressed[6::8] = compressed[:] >> 1 & 1
  uncompressed[7::8] = compressed[:] & 1

  return uncompressed


def load_gt_volume(filename):
  basename = os.path.splitext(filename)[0]

  labels = np.fromfile(filename, dtype=np.uint16)
  invalid_voxels = unpack(np.fromfile(basename + ".invalid", dtype=np.uint8))

  return labels, invalid_voxels

def load_pred_volume(filename):
  labels = np.fromfile(filename, dtype=np.uint16)
  return labels

# possible splits
splits = ["train", "valid", "test"]

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="SSC semantic-kitti")

  parser.add_argument(
      '--dataset', '-d',
      type=str,
      required=False,
      default='/home/wenzhao/datasets/SemanticKITTI/dataset/',
      help='Dataset dir. No Default',
  )

  parser.add_argument(
      '--predictions', '-p',
      type=str,
      required=False,
      help='Prediction dir. Same organization as dataset, but predictions in'
      'each sequences "prediction" directory.'
  )
  parser.add_argument(
      '--datacfg', '-dc',
      type=str,
      required=False,
      default="kitti_ssc/data/semantic_kitti/semantic-kitti.yaml",
      help='Dataset config file. Defaults to %(default)s',
  )

  parser.add_argument(
      '--split', '-s',
      type=str,
      required=False,
      choices=["train", "valid", "test"],
      default="valid",
      help='Split to evaluate on. One of ' +
      str(splits) + '. Defaults to %(default)s',
  )
  parser.add_argument(
      '--output',
      dest='output',
      type=str,
      default=".",
      help='Exports "scores.txt" to given output directory for codalab'
      'Defaults to %(default)s',
  )

  args = parser.parse_args()
  print("  ========================== Arguments ==========================  ")
  print("\n".join(["  {}:\t{}".format(k,v) for (k,v) in vars(args).items()]))
  print("  ===============================================================  \n")
  gt_data_root = args.dataset

  DATA = yaml.safe_load(open(args.datacfg, 'r'))

  # get number of interest classes, and the label mappings
  class_strings = DATA["labels"]
  class_remap = DATA["learning_map"]
  class_inv_remap = DATA["learning_map_inv"]
  class_ignore = DATA["learning_ignore"]
  n_classes = len(class_inv_remap)

  test_sequences = DATA["split"][args.split]

  # make lookup table for mapping
  maxkey = max(class_remap.keys())

  # +100 hack making lut bigger just in case there are unknown labels
  remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
  remap_lut[list(class_remap.keys())] = list(class_remap.values())

  # in completion we have to distinguish empty and invalid voxels.
  # Important: For voxels 0 corresponds to "empty" and not "unlabeled".
  remap_lut[remap_lut == 0] = 255   # map 0 to 'invalid'
  remap_lut[0] = 0  # only 'empty' stays 'empty'.

  from kitti_ssc.scripts.np_ioueval import iouEval
  evaluator = iouEval(n_classes, [])

  # get files from ground truth and predictions.
  filenames_gt = []
  filenames_pred = []
  for seq in test_sequences:
    seq_dir_gt = os.path.join("sequences", '{0:02d}'.format(int(seq)), "voxels")
    seq_dir_pred = os.path.join("sequences", '{0:02d}'.format(int(seq)), "predictions")

    gt_file_list = [f for f in os.listdir(os.path.join(args.dataset, seq_dir_gt)) if f.endswith(".label")]
    filenames_gt.extend([os.path.join(seq_dir_gt, f) for f in gt_file_list])
    filenames_pred.extend([os.path.join(seq_dir_pred, f) for f in gt_file_list])

  missing_pred_files = False

  if args.predictions is None:
    prediction_dir = args.dataset
  else:
    prediction_dir = args.predictions

  # check that all prediction files exist
  for pred_file in filenames_pred:
    if not os.path.exists(os.path.join(prediction_dir, pred_file)):
      print("Expected to have {}, but file does not exist!".format(pred_file))
      missing_pred_files = True

  if missing_pred_files: raise RuntimeError("Error: Missing prediction files! Aborting evaluation.")

  evaluation_pairs = list(zip(filenames_gt, filenames_pred))

  print("Evaluating: ", end="", flush=True)
  progress = 10

  for i, f in enumerate(evaluation_pairs):
    if 100.0 * i / len(evaluation_pairs) >= progress:
      print("{}% ".format(progress), end="", flush=True)
      progress = progress + 10

    filename_gt = os.path.join(args.dataset, f[0])
    filename_pred = os.path.join(prediction_dir, f[1])

    pred = load_pred_volume(filename_pred)
    target, invalid_voxels = load_gt_volume(filename_gt)

    # Map labels "pred_labels" and "gt_labels" from semantic-kitti ID's to [0 : n_classes -1]
    pred = remap_lut[pred]
    target = remap_lut[target]

    masks = get_eval_mask(target, invalid_voxels)

    target = target[masks]
    pred = pred[masks]

    # add single scan to evaluation
    evaluator.addBatch(pred, target)
  print("Done \U0001F389.")
  print("\n  ========================== RESULTS ==========================  ")
  # when I am done, print the evaluation
  _, class_jaccard = evaluator.getIoU()
  m_jaccard = class_jaccard[1:].mean()


  print('Validation set:\nIoU avg {m_jaccard:.3f}'.format(m_jaccard=m_jaccard))
  ignore = [0]
  # print also classwise
  for i, jacc in enumerate(class_jaccard):
    if i not in ignore:
      print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
          i=i, class_str=class_strings[class_inv_remap[i]], jacc=jacc))

  # compute remaining metrics.
  conf = evaluator.get_confusion()
  precision = np.sum(conf[1:,1:]) / (np.sum(conf[1:,:]) + epsilon)
  recall = np.sum(conf[1:,1:]) / (np.sum(conf[:,1:]) + epsilon)
  acc_cmpltn = (np.sum(conf[1:, 1:])) / (np.sum(conf) - conf[0,0])
  mIoU_ssc = m_jaccard

  print("Precision =\t" + str(np.round(precision * 100, 2)) + '\n' +
        "Recall =\t" + str(np.round(recall * 100, 2)) + '\n' +
        "IoU Cmpltn =\t" + str(np.round(acc_cmpltn * 100, 2)) + '\n' +
        "mIoU SSC =\t" + str(np.round(mIoU_ssc * 100, 2)))

  # write "scores.txt" with all information
  results = {}
  results["iou_completion"] = float(acc_cmpltn)
  results["iou_mean"] = float(mIoU_ssc)

  for i, jacc in enumerate(class_jaccard):
    if i not in ignore:
      results["iou_"+class_strings[class_inv_remap[i]]] = float(jacc)

  output_filename = os.path.join(args.output, 'scores.txt')
  with open(output_filename, 'w') as yaml_file:
    yaml.dump(results, yaml_file, default_flow_style=False)