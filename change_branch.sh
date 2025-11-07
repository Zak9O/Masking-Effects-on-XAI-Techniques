#!/bin/sh
# This script switches between 'main' and 'annonymization' branches.
# Before switching, it removes the .venv directory and runs 'uv sync'.

# Exit immediately if a command exits with a non-zero status.
set -e

# Get the current branch name
current_branch=$(git branch --show-current)

main_branch="main"
anon_branch="annonymization"

if [ "$current_branch" = "$main_branch" ]; then
  echo "On branch '$main_branch'. Preparing to switch to '$anon_branch'."
  echo "Switching to branch '$anon_branch'..."
  git checkout "$anon_branch"
  echo "Switched to branch '$anon_branch'."
  echo "Running uv sync..."
  uv sync
elif [ "$current_branch" = "$anon_branch" ]; then
  echo "On branch '$anon_branch'. Preparing to switch to '$main_branch'."
  echo "Switching to branch '$main_branch'..."
  git checkout "$main_branch"
  echo "Switched to branch '$main_branch'."
  echo "Running uv sync..."
  uv sync
else
  echo "Error: Not on branch '$main_branch' or '$anon_branch'. Currently on '$current_branch'."
  echo "No action taken."
  exit 1
fi

echo "Branch switch complete."
