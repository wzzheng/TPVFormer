#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import zipfile
import argparse
import os
import numpy as np


class ValidationException(Exception):
  pass

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

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Validate a submission zip file needed to evaluate on CodaLab competitions.\n\nThe verification tool checks:\n  1. correct folder structure,\n  2. existence of label files for each scan,\n  3. count of labels for each scan.\nInvalid labels are ignored by the evaluation script, therefore we don't check\nfor invalid labels.", formatter_class=argparse.RawTextHelpFormatter)

  parser.add_argument(
      "zipfile",
      type=str,
      help='zip file that should be validated.',
  )

  parser.add_argument(
      'dataset',
      type=str,
      help='directory containing the folder "sequences" containing folders "11", ..., "21" with the input data ("velodyne" or "voxels") folder.'
  )

  parser.add_argument(
      "--task",
      type=str,
      choices=["segmentation", "completion", "panoptic"],
      default="segmentation",
      help='task for which the zip file should be validated.'
  )

  FLAGS, _ = parser.parse_known_args()

  checkmark = "\u2713"

  float_bytes = 4
  uint32_bytes = 4
  uint16_bytes = 2

  try:

    print('Validating zip archive "{}".\n'.format(FLAGS.zipfile))

    print( " ============ {:^10} ============ ".format(FLAGS.task))

    print("  1. Checking filename.............. ", end="", flush=True)
    if not FLAGS.zipfile.endswith('.zip'):
      raise ValidationException('Competition bundle must end with ".zip"')
    print(checkmark)

    with zipfile.ZipFile(FLAGS.zipfile) as zipfile:
      if FLAGS.task == "segmentation" or FLAGS.task == "panoptic":
        

        print("  2. Checking directory structure... ", end="", flush=True)

        directories = [folder.filename for folder in zipfile.infolist() if folder.filename.endswith("/")]
        if "sequences/" not in directories:
          raise ValidationException('Directory "sequences" missing inside zip file.')

        for sequence in range(11, 22):
          sequence_directory = "sequences/{}/".format(sequence)
          if sequence_directory not in directories:
            raise ValidationException('Directory "{}" missing inside zip file.'.format(sequence_directory))
          predictions_directory = sequence_directory + "predictions/"
          if predictions_directory not in directories:
            raise ValidationException('Directory "{}" missing inside zip file.'.format(predictions_directory))

        print(checkmark)

        print('  3. Checking file sizes............ ', end='', flush=True)

        prediction_files = {info.filename: info for info in zipfile.infolist() if not info.filename.endswith("/")}

        for sequence in range(11, 22):
          sequence_directory = 'sequences/{}'.format(sequence)
          velodyne_directory = os.path.join(FLAGS.dataset, 'sequences/{}/velodyne/'.format(sequence))

          velodyne_files = sorted([os.path.join(velodyne_directory, file) for file in os.listdir(velodyne_directory)])
          label_files = sorted([os.path.join(sequence_directory, "predictions", os.path.splitext(filename)[0] + ".label")
                                for filename in os.listdir(velodyne_directory)])

          for velodyne_file, label_file in zip(velodyne_files, label_files):
            num_points = os.path.getsize(velodyne_file) / (4 * float_bytes)

            if label_file not in prediction_files:
              raise ValidationException('"' + label_file + '" is missing inside zip.')

            num_labels = prediction_files[label_file].file_size / uint32_bytes
            if num_labels != num_points:
              raise ValidationException('label file "' + label_file +
                                        "' should have {} labels, but found {} labels!".format(int(num_points), int(num_labels)))

        print(checkmark)
      elif FLAGS.task == "completion":
        print("  2. Checking directory structure... ", end="", flush=True)

        directories = [folder.filename for folder in zipfile.infolist() if folder.filename.endswith("/")]
        if "sequences/" not in directories:
          raise ValidationException('Directory "sequences" missing inside zip file.')

        for sequence in range(11, 22):
          sequence_directory = "sequences/{}/".format(sequence)
          if sequence_directory not in directories:
            raise ValidationException('Directory "{}" missing inside zip file.'.format(sequence_directory))
          predictions_directory = sequence_directory + "predictions/"
          if predictions_directory not in directories:
            raise ValidationException('Directory "{}" missing inside zip file.'.format(predictions_directory))

        print(checkmark)

        print('  3. Checking file sizes', end='', flush=True)

        prediction_files = {str(info.filename): info for info in zipfile.infolist() if not info.filename.endswith("/")}
        
        # description.txt is optional and one should not get an error.
        if "description.txt" in prediction_files: del prediction_files["description.txt"]


        necessary_files = []

        for sequence in range(11, 22):

          sequence_directory = 'sequences/{}'.format(sequence)
          voxel_directory = os.path.join(FLAGS.dataset, 'sequences/{}/voxels/'.format(sequence))

          voxel_files = sorted([os.path.join(voxel_directory, file) for file in os.listdir(voxel_directory) if file.endswith(".bin")])
          label_files = sorted([os.path.join(sequence_directory, "predictions", os.path.splitext(filename)[0] + ".label")
                                for filename in os.listdir(voxel_directory)])
          necessary_files.extend(label_files)

          for voxel_file, label_file in zip(voxel_files, label_files):
            input_voxels = unpack(np.fromfile(voxel_file, dtype=np.uint8))
            num_voxels = input_voxels.shape[0] # fixed volume (= 256 * 256 * 32)!

            if label_file not in prediction_files:
              raise ValidationException('"' + label_file + '" is missing inside zip.')

            num_labels = prediction_files[label_file].file_size / uint16_bytes # expecting uint16 for labels.
            if num_labels != num_voxels:
              raise ValidationException('label file "' + label_file +
                                        "' should have {} labels, but found {} labels!".format(int(num_voxels), int(num_labels)))
          print(".", end="", flush=True)
        print(". ", end="", flush=True)
        print(checkmark)

        print('  4. Checking for unneeded files', end='', flush=True)
        if len(necessary_files) != len(prediction_files.keys()):
          filelist = sorted([f for f in prediction_files.keys() if f not in necessary_files])
          ell = ""
          if len(filelist) > 10: ell = ", ..."
          raise ValidationException("Zip contains unneeded predictions, e.g., {}".format(",".join(filelist[:10]) + ell))
        
        print(".... " + checkmark)
      else:
        raise NotImplementedError("Unknown task.")
  except ValidationException as ex:
    print("\n\n  " + "\u001b[1;31m>>> Error: " + str(ex) + "\u001b[0m")
    exit(1)

  print("\n\u001b[1;32mEverything ready for submission!\u001b[0m  \U0001F389")