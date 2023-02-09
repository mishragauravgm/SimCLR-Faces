squeue -u mishra.g

srun --partition=gpu --nodes=8 --pty --gres=gpu:p100:1 --ntasks=1 --mem=128GB --time=08:00:00 /bin/bash



31051760