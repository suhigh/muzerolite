#!/bin/bash

app_dir=~/dockerappsrc

if [ ! -d $app_dir ]
then
  mkdir -p $app_dir
fi

rm -fr $app_dir/*

cp ./*.py $app_dir
mkdir $app_dir/protos
cp ./protos/*.* $app_dir/protos/
cp ./tictactoe/*.py $app_dir

echo "project has been build to $app_dir, please link this folder to docker path!"