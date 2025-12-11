import argparse

from sklearn.model_selection import train_test_split


def read_sysevr_dataset(path):
    with open(path, 'r') as file:
        lines = file.readlines()
        records = []
        start = 0
        for idx, line in enumerate(lines):
            if line.startswith('---'):
                records.append((start, idx + 1))
                start = idx + 1
    return lines, records


def save(filename, lines, records):
    with open(filename, 'w') as file:
        for record in records:
            file.writelines(lines[record[0]:record[1]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split SySeVR dataset')
    parser.add_argument('--file', type=str, default=None, metavar='path',
                        required=True, help='input data file')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()

    lines, records = read_sysevr_dataset(args.file)

    # 0.6/0.2/0.2 split
    records_train, records_test = train_test_split(records, test_size=0.2, random_state=args.seed)
    records_train, records_valid = train_test_split(records_train, test_size=0.25, random_state=args.seed)

    save('train.txt', lines, records_train)
    save('test.txt', lines, records_test)
    save('valid.txt', lines, records_valid)
