import sys
import os
import matplotlib.pyplot as plt
import re

def parse_log_file(filename):
    """
    Parses the log file and returns two lists:
    - x: number of sentences processed (divided by 100)
    - y: dev accuracy at each checkpoint
    """
    x = []
    y = []
    if not os.path.exists(filename):
        return x, y
    with open(filename) as f:
        for line in f:
            # Example: After 500 sentences: Train Accuracy = 0.9876, Dev Accuracy = 0.9876, ...
            m = re.search(r"After (\d+) sentences:.*Dev Accuracy = ([0-9.]+)", line)
            if m:
                num_sent = int(m.group(1))
                dev_acc = float(m.group(2))
                x.append(num_sent / 100)
                y.append(dev_acc)
    return x, y

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ['pos', 'ner']:
        print("Usage: python3 plot-graph.py <pos|ner>")
        sys.exit(1)
    task = sys.argv[1]
    folder = os.path.dirname(__file__)
    modes = ['a', 'b', 'c', 'd']
    mode_labels = ['Mode a', 'Mode b', 'Mode c', 'Mode d']
    colors = ['blue', 'orange', 'green', 'red']
    lines = []
    for i, mode in enumerate(modes):
        log_file = os.path.join(folder, f'log_{mode}_{task}.txt')
        if not os.path.exists(log_file):
            log_file = os.path.join(folder, f'log_hidim_{mode}_{task}.txt')
        x, y = parse_log_file(log_file)
        if x and y:
            plt.plot(x, y, label=mode_labels[i], color=colors[i])
            lines.append(True)
        else:
            lines.append(False)
    plt.xlabel('Num sentences / 100')
    plt.ylabel('Dev Accuracy')
    plt.title(f'Dev Accuracy vs Num Sentences ({task.upper()})')
    plt.legend()
    plt.grid(True)
    if any(lines):
        plt.show()
    else:
        print("No log files found for any mode.")

if __name__ == "__main__":
    main()
