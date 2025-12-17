#!/bin/sh
# Use the following command to delete all folders called data in the tmp file
# find . -type d -name "data" -exec rm -rf {} +
# # Use to delete all t_closeness files called 1.0
# find . -type f -path "*/t_closeness/1.0.csv.npy" -delete
rm -rf ./tmp
scp -i ~/.ssh/id_ed25519 -r s225169@transfer.gbar.dtu.dk:explanation/out/ tmp
cd tmp
find . -type d -name "data" -exec rm -rf {} +
find . -type f -path "*/t_closeness/1.0.csv.npy" -delete

cd ..

# The original source directory (renamed to tmp)
SRC_DIR="$WHOME/Downloads/out"

# 1. Create the 'tmp' working directory from the source
# cp -r "$SRC_DIR" tmp

# 2. Define the new parent directory outside of 'tmp'
DEST_ROOT="../data"

# Create the destination root directory if it doesn't exist
mkdir -p "$DEST_ROOT"

# 3. Find all dataset directories (children of 'lime' or 'shap') under 'tmp'
find tmp -type d \( -name "lime" -o -name "shap" \) -print0 | while IFS= read -r -d $'\0' EXPLAINER_DIR; do
    # Iterate over every dataset directory *inside* the explainer folder
    for DATASET_DIR in "${EXPLAINER_DIR}"/*; do

        if [ -d "$DATASET_DIR" ]; then
            # DATASET_DIR example: tmp/forest/lime/adult
            
            # Construct the relative path (forest/lime/adult)
            REL_PATH="${DATASET_DIR#tmp/}"
            
            # Construct the final destination path (../data/forest/lime/adult)
            DEST_PATH="$DEST_ROOT/$REL_PATH"
            

            # 💡 CRITICAL CHANGE: Remove the old target directory completely before moving the new one.
            if [ -d "$DEST_PATH" ]; then
                rm -rf "$DEST_PATH"
            fi

            
            # Move the new dataset directory into the new structure.

            # This is a simple move, not a content merge.
            mv "$DATASET_DIR" "$DEST_ROOT/$REL_PATH"
        fi
    done
done


# 4. Cleanup the empty 'tmp' directories left behind.
# find tmp -type d -empty -delete
