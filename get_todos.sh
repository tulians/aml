#!/usr/bin/env bash

function join_by { local IFS="$1"; shift; echo "$*"; }

TODOS=$(grep -R 'TODO:' ./src)

if [[ $? -eq 0 ]];
then
  echo "List of TODOS:"
  string=$(join_by ' ' ${TODOS[@]})

  separator='./'
  printf '%s\n' "${string//$separator/$'\n'}"

else
  echo "There are no pending TODOs."
fi
