#!/usr/bin/env bash

# Set default values
color="yellow"
session="mysession"

# Parse command line options
while getopts ":c:s:" opt; do
  case $opt in
    c)
      color="$OPTARG"
      ;;
    s)
      session="$OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

# Set the status bar color
tmux set -g status-style bg=${color}

# Start tmux
tmux new-session -s ${session}

