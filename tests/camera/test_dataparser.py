import sys
sys.path.append('../../')

from pointrix.dataset.colmap_data import ColmapReFormat
from pointrix.dataset.base_data import BaseDataPipline

def test_ReFormat():
    colmap = ColmapReFormat(cfg=None, data_root='/home/clz/code_remote/Pointrix/pointrix/data/truck')
    print(colmap.data_list)
    print(len(colmap.data_list))

def test_Pipline():
    colmap_pipline = BaseDataPipline(config=None, path='/home/clz/code_remote/Pointrix/pointrix/data/truck')
    for i in range(100):
        print(colmap_pipline.next_train(i)['camera']['idx'])

if __name__ == "__main__":
    test_ReFormat()
    test_Pipline()