#!/bin/bash
# Run this script using the following command:
# bash convert_datasets.sh
script_dir=$(dirname "$(readlink -f "$0")")

data_root="${script_dir}/datafolder/raw_data"
records_root="${script_dir}/datafolder/converted_data/"

declare -a all_sources=("colon_crc_tp" "colon_kather19" "colon_kbsmc" "etc_lc25000" "gastric_kbsmc" "prostate_panda" "breast_bach")

#TFrecord process
for source in "${all_sources[@]}"
do
    echo "Processing ${source}..."
    python3 create_records.py --data_root "${data_root}" --records_root "${records_root}" --name "${source}"
done
echo "All datasets processed."


#Indices process
for source in "${all_sources[@]}"
do
    echo "Processing ${source}..."
    source_path="${records_root}/${source}"
    find "${source_path}" -name '*.tfrecords' -type f -exec sh -c 'python3 -m fewshot_lib.tfrecord2idx "$1" "${1%.tfrecords}.index"' sh {} \;
done
echo "All indices processed."
