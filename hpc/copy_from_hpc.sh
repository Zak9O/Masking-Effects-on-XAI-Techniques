#!/bin/sh
mv $WHOME/Downloads/out/ tmp
# scp -r -i ~/.ssh/id_ed25519 "s225169@transfer.gbar.dtu.dk:~/explanation/out/*" tmp/

datasets=("adult" "usa_house" "cervic_cancer")

for dataset in "${datasets[@]}"; do

  rm -rf ../data/${dataset}/lime/ ../data/${dataset}/shap/
  mkdir ../data/${dataset}/lime/
  mkdir ../data/${dataset}/shap/
  cp -r "./tmp/lime/data/${dataset}/"* "../data/${dataset}/lime/"
  cp -r "./tmp/shap/data/${dataset}/"* "../data/${dataset}/shap/"
done

rm -rf tmp

