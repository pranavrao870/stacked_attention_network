import sys

from ocr import process_images
from ocr import sort_by_pos

def test_box_detection(checkpoint_path, crnn_path):
    process_images('test_data', ['train'], [range(1, 201)], checkpoint_path, crnn_path)

def test_sort():
    t = [(100, 0), (90, 10), (80, 20), (70, 35), (60, 10), (50, 10)]
    t1 = [t[3], t[2], t[1], t[5], t[0], t[4]]
    im = ['a', 'b', 'c', 'd', 'e', 'f']
    print(sort_by_pos(im, im, t1, (100, 0, 0)))

if __name__ == '__main__':
    #test_sort()
    test_box_detection(sys.argv[1], sys.argv[2])
