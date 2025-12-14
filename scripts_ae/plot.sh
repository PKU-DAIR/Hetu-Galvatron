if [ $# -ne 2 ]; then
    echo "Usage: $0 <figure_id> <type>"
    exit 1
fi

figure_id=$1
type=$2

if [ "$type" == "default" ]; then
    python scripts_ae/${figure_id}_plot.py default
fi

if [ "$type" == "new" ]; then
    if [ "$figure_id" != "11" ]; then
        python scripts_ae/${figure_id}_analysis.py
    fi
    python scripts_ae/${figure_id}_plot.py
fi