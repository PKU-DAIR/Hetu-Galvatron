if [ $# -ne 1 ]; then
    echo "Usage: $0 <type>"
    exit 1
fi

type=$1

if [ "$type" == "default" ]; then
    python scripts_ae/10b_plot.py default
fi

if [ "$type" == "new" ]; then
    python scripts_ae/10b_analysis.py
    python scripts_ae/10b_plot.py
fi