import pandas as pd

from HorizonAnneal import HorizonAnneal
from penalty_spec__Riley_62_set import produce_list
from timer import Timer


def print_hi(name):
    print(f'Hi, {name}')


if __name__ == '__main__':
    t = Timer(text="Total time: {:.5f} seconds")
    t.start()
    myCH = pd.read_csv("./dataset/Horizon_Data_srandom_for_HA.csv")
    mySections = pd.read_csv("./dataset/riley_section_names.csv")
    myCH = myCH.drop(["Section_No"], axis=1)
    penalty_spec_62 = produce_list()
    j = HorizonAnneal(myCH, pen_str=penalty_spec_62)
    t.stop()

