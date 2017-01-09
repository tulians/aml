#!/usr/bin/env bash

description=$1

git add .
git status

echo $description

echo "Is the commit description correct? [y/n]:"
read prompt

if [[ ("$prompt" == "Y") || ("$prompt" == "y") ]];
then
  git commit -m "$description"
  git push origin master
else
  echo "Run the command again with the correct commit description."
fi
