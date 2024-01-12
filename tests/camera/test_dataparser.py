import sys
sys.path.append('../../')

from pointrix.dataset.colmap_data import ColmapReFormat

def test_ReFormat():
    colmap = ColmapReFormat(cfg=None, data_root='/home/clz/code_remote/Pointrix/pointrix/data/truck')
    print(colmap.data_list)
    print(colmap.data_list[0])
    print(len(colmap.data_list))

if __name__ == "__main__":
    test_ReFormat()