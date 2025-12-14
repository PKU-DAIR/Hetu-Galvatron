if [ $# -eq 2 ]; then
    python scripts_ae/planner.py $1 $2
else
    python scripts_ae/planner.py
fi