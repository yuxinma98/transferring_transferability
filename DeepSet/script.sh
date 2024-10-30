task_id=(1 2 3 4)
for task in "${task_id[@]}"; do
    echo "Running main.py with task $task"
    CUDA_VISIBLE_DEVICES=1 python main.py --task_id "$task"
done