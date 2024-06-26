import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    general = parser.add_argument_group(title='general')
    general.add_argument('-data', type=str, default='', help='input dataset name')
    general.add_argument('-out', type=str, default='../result/ssumm/', help='output folder')
    general.add_argument('-k', type=float, default=0.8, help='fracK')
    general.add_argument('-re', type=int, default=2, help='compute reconstruction error, RE1 or RE2 ?')
    general.add_argument('-seed', type=int, default=2023, help='random seed')
    general.add_argument('-use_fast_topndrop', type=int, default=0,
                         help='use_fast_topndrop with median search (use_fast_topndrop=1) or just sort (use_fast_topndrop=0) ?')
    general.add_argument('-sidx', type=int, default=0,
                         help='save result with new index (sidx=1) or initial node id (sidx=0) ?')
    general.add_argument('-T', type=int, default=20,
                         help='number of iteration')
    general.add_argument('-cals', type=int, default=10,
                         help='How many times do we calculate shingle?')
    ################################################
    general.add_argument('-urd', type=int, default=0,
                         help='user defined random function')
    general.add_argument('-threshold_start', type=int, default=1,
                         help='threshold_start')
    ################################################
    general.add_argument('-useNE', type=int, default=0,# flag args
                         help='use node embedding')
    general.add_argument('-uselazy', type=int, default=0,
                         help='use lazy!!!')
    general.add_argument('-NEmodel', type=str, default='',
                         help='NE model')

    # 6.26
    general.add_argument('-use_label', type=int, default=0,  # flag args  store true!!!!!
                         help='use node embedding')


    args, _ = parser.parse_known_args()
    return args