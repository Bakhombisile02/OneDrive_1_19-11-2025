#!/bin/bash
# Convenience script for Stacey's workflow
# Runs Task 3a (MI + Elastic Net) and Task 4a (Soft Voting Ensemble)

echo "Starting Stacey's Workflow..."
python assignment.py --flow stacey
echo "Done! Check the 'results/' directory for output CSVs."
