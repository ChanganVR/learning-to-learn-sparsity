import os
import shutil

os.mkdir('datasets/dtd_splits')

for phase in ['train', 'val', 'test']:
    label_file = 'datasets/dtd_orig/labels/{}1.txt'.format(phase)
    split_folder = 'datasets/dtd_splits/{}'.format(phase)
    os.mkdir(split_folder)
    with open(label_file) as fo:
        files = [x.strip() for x in fo.readlines()]
        for file in files:
            category = os.path.dirname(file)
            if not os.path.exists(os.path.join(split_folder, category)):
                os.mkdir(os.path.join(split_folder, category))
            shutil.copyfile('datasets/dtd_orig/images/'+file, os.path.join(split_folder, file))
