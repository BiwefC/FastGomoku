rm ../data/raw/* -rf

while [ 1 ]
do
python3 evaluator.py
python3 selfplay.py
python3 train.py
if [ $? -ne 0 ]; then
    echo "fail"
    break
else
    echo "success"
fi
done