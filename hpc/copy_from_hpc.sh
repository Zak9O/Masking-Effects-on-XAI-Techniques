#!/bin/sh
# mkdir -p tmp
# scp -r -i ~/.ssh/id_ed25519 "s225169@transfer.gbar.dtu.dk:~/anonymization/out/*" tmp/

for dataset_folder in tmp/*; do
    if [ -d "$dataset_folder" ]; then
        dataset_name=$(basename "$dataset_folder")
        
        for method_folder in "$dataset_folder"/*; do
            if [ -d "$method_folder" ]; then
                method_name=$(basename "$method_folder")
                
                target_path="../data/$dataset_name/$method_name"
                
                if [ -d "$target_path" ]; then
                    echo "Deleting existing $target_path"
                    rm -rf "$target_path"
                fi
                
                echo "Moving $method_folder to ../data/$dataset_name/"
                mv "$method_folder" "../data/$dataset_name/"
            fi
        done
    fi
done

rm -rf tmp

