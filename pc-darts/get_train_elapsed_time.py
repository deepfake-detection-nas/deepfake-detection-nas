import os
import torch
import re
from datetime import timedelta

def read_state_file(filepath):
    try:
        state = torch.load(filepath)
        # print(state.keys())
        if 'train_elapsed_time' in state:
            return state['train_elapsed_time']
        else:
            return None
    except:
        return None

def parse_time(time_str):
    if ':' in time_str:
        h, m, s = map(int, time_str.split(':'))
        return timedelta(hours=h, minutes=m, seconds=s)
    else:
        return timedelta(seconds=int(time_str))

def sum_training_times(log_files):
    total_time = timedelta()
    time_pattern = re.compile(r'.+\[=*\]\s+-\s+(\d+:\d+:\d+|\d+)s?,')
    gpu_count_pattern = re.compile(r'ALL GPU COUNT: (\d+)')
    for file in log_files:
        with open(file, 'r') as f:
            lines = f.readlines()
            gpu_count = 1  # Default GPU count
            for line in lines:
                gpu_match = gpu_count_pattern.search(line)
                if gpu_match:
                    gpu_count = int(gpu_match.group(1))
                    continue
                time_match = time_pattern.search(line)
                if time_match:
                    # print(time_match.group(1))
                    time = parse_time(time_match.group(1))
                    adjusted_time = time * gpu_count
                    total_time += adjusted_time
    return total_time

def main(directory):
    for subdir in sorted(os.listdir(directory)):
        if subdir.startswith('search-CELEB'):
            print("CHECK: "+subdir)
            fin_flg_file = os.path.join(directory, subdir, 'finish_training')
            if not os.path.isfile(fin_flg_file):
                print("\tDoes not finish.")
                continue
            state_file = os.path.join(directory, subdir, 'state.pt')
            train_time = read_state_file(state_file)
            if train_time is not None:
                print(f"\tTraining time in {subdir}: {train_time}")
            else:
                log_files = [os.path.join(directory, subdir, f) for f in os.listdir(os.path.join(directory, subdir)) if re.match(r'.*\.sh\.o.*', f)]
                total_time = sum_training_times(log_files)
                print(f"\tTraining time in {subdir}: {total_time}  (calculated from logs)")

if __name__ == "__main__":
    main('./')
