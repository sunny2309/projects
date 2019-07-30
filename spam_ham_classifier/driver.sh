#!/usr/bin/env bash

set -e

SLEEP_TIME=0s

for data_dir in $(ls -d /home/sunny/Downloads/spam-ham/train_*); do  ### Modify this location pointing towards where you have put your datasets.
    echo '-------------------------------------------------------------'
    echo "Remove hammie.db. Start training afresh for this new dataset."
    rm -f hammie.db
    #echo "Train: $(basename $data_dir)"
    var1=$(echo $(basename $data_dir) | cut -f2 -d_)
    echo $var1
    echo '-------------------------------------------------------------'

    #if [[ $(basename $data_dir) = 'train_trec07p' ]]; then
	#continue
    #fi
    
    if [[ $(basename $data_dir) = 'train_PU1' ]]; then
	continue
    fi
    
    if [[ $(basename $data_dir) = 'train_ENRON' ]]; then
        data_dir="${data_dir}/enron"
        for data_dir in ${data_dir}{1..6}; do
            test_dir=${data_dir/train/test}
            echo "Train dir: $data_dir, Test dir: $test_dir"
            python2 hammie/hammiebulk.py -d hammie.db -g "${data_dir}/h" \
                          -s "${data_dir}/s"                          \
                          -a "${test_dir}/h"                          \
                          -b "${test_dir}/s"               
            echo "Sleeping for ${SLEEP_TIME} minutes. Let it cool down a bit."
            sleep ${SLEEP_TIME}
        done
    else

        test_dir=${data_dir/train/test}
        echo "Train dir: $data_dir, Test dir: $test_dir"
        python2 hammie/hammiebulk.py -d hammie.db -g "${data_dir}/h" \
                -s "${data_dir}/s"                          \
                -a "${test_dir}/h"                          \
                -b "${test_dir}/s"              

        echo "Sleeping for ${SLEEP_TIME} minutes. Let it cool down a bit."
        sleep ${SLEEP_TIME}

    fi
done

