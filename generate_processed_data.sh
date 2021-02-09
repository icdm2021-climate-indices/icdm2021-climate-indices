#!/bin/bash

SAVEDIR="reproducibility_check/"
WESTUSDIR="masked_data_western_us/"
PACDIR="masked_data_pacific/"

for split in "train" "test";
    do
    python aggregate_data.py --variable tmp2m \
                             --load_directory "${WESTUSDIR}" \
                             --split "${split}" \
                             --save_directory "${SAVEDIR}" \
                             --save_name "anomalies_tmp2m_western_us_14_${split}.h5" \
                             --number_of_days 14 \
                             --mean_std_file tmp2m_western_us_14_mean_std.h5

    python aggregate_data.py --variable tmp2m \
                             --load_directory "${WESTUSDIR}" \
                             --split "${split}" \
                             --save_directory "${SAVEDIR}" \
                             --save_name "anomalies_tmp2m_western_us_14_${split}_no_standardize.h5" \
                             --number_of_days 14 \
                             --mean_std_file tmp2m_western_us_14_mean_std.h5 \
                             --no_standardize
    done

for climvar in "slp" "sst" "hgt500" "rhum.sig995";
    do
    for split in "train" "test";
        do
        python aggregate_data.py --variable "${climvar}" \
                                 --load_directory "${PACDIR}" \
                                 --split "${split}" \
                                 --save_directory "${SAVEDIR}" \
                                 --save_name "anomalies_${climvar}_pacific_14_${split}.h5" \
                                 --number_of_days 14 \
                                 --mean_std_file "${climvar}_pacific_14_mean_std.h5"
        done
    done
