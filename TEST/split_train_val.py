import os


if __name__ == '__main__':
    BASE_DIR = r'D:\DATASET\CUB_200_2011\CUB_200_2011'
    images_path = os.path.join(BASE_DIR, 'images.txt')
    splits_path = os.path.join(BASE_DIR, 'train_test_split.txt')
    train_txt_path = os.path.join(BASE_DIR, 'train.txt')
    val_txt_path = os.path.join(BASE_DIR, 'val.txt')

    with open(images_path, 'r') as f:
        lines = f.readlines()
        img_names = [
            str(int(line.split()[1].split('.')[0]) - 1)
            + ' ' + line.strip().split()[1] for line in lines
        ]

    with open(splits_path, 'r') as f:
        lines = f.readlines()
        splits = [line.strip().split()[1] for line in lines]

    trains, vals = [], []
    for i in range(len(img_names)):
        if splits[i] == '1':
            trains.append(img_names[i])
        else:
            vals.append(img_names[i])

    with open(train_txt_path, 'w+') as f:
        f.write('\n'.join(trains))

    with open(val_txt_path, 'w+') as f:
        f.write('\n'.join(vals))
