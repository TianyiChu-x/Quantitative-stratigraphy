import pandas as pd
import numpy as np
import csv

from HorizonAnneal import HorizonAnneal
from penalty_spec__Riley_62_set import produce_list
from timer import Timer


def print_hi(name):
    print(f'Hi, {name}')


if __name__ == '__main__':
    t = Timer(text="Total time: {:.5f} seconds")
    t.start()
    myCH = pd.read_csv("./dataset/riley_62_for_R.csv")
    mySections = pd.read_csv("./dataset/riley_section_names.csv")
    myCH = myCH.drop(["Section_Name"], axis=1)
    penalty_spec_62 = produce_list()
    j = HorizonAnneal(myCH, pen_str=penalty_spec_62)
    # 将ndarray写入CSV文件
    with open("output.csv", "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in j:
            csv_writer.writerow(row)
    t.stop()

