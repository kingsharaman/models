#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

CURRENT_DIR=$(pwd)
WORK_DIR="./pascal_voc_seg"

cd "${CURRENT_DIR}"

# Root path for PASCAL VOC 2012 dataset.
PASCAL_ROOT="${WORK_DIR}/VOCdevkit/VOC2012"

# Remove the colormap in the ground truth annotations.
SEMANTIC_SEG_FOLDER="${PASCAL_ROOT}/PersonSegmentationRaw"

# Build TFRecords of the dataset.
# First, create output directory for storing TFRecords.
OUTPUT_DIR="${WORK_DIR}/tfrecord_person"
mkdir -p "${OUTPUT_DIR}"

IMAGE_FOLDER="${PASCAL_ROOT}/JPEGImages"
LIST_FOLDER="${PASCAL_ROOT}/ImageSets/PersonSegmentation"

echo "Converting PASCAL VOC 2012 dataset..."
python ./build_voc2012_data.py \
  --image_folder="${IMAGE_FOLDER}" \
  --semantic_segmentation_folder="${SEMANTIC_SEG_FOLDER}" \
  --list_folder="${LIST_FOLDER}" \
  --image_format="jpg" \
  --output_dir="${OUTPUT_DIR}"
