if [ $# -ne 1 ]; then
    echo "Usage: $0 <type>"
    exit 1
fi

type=$1

if [ "$type" == "default" ]; then
    python scripts_ae/11_plot.py default
fi

if [ "$type" == "new" ]; then
    python scripts_ae/11_plot.py
fi