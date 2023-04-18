import math
import random
import numpy as np

from ColumnRangeExtension import NetRangeExtension, ftransitions2
from timer import Timer


def HorizonAnneal(dataframe, pen_str, nouter=400, ninner=100, temperature=5, cooling=0.5):
    global pcv3, j, pmove, psec
    d3 = dataframe.values  # 取出dataframe中的数值 -> ndarray
    columns = dataframe.columns  # 取出表头留作备用
    data_offset = 4  # 数据偏移量

    # 引入计时
    ptime = Timer(text="HA penalty time: {:.5f} seconds")
    ptime.start()

    # 根据score对数据进行重新排序
    d3ord = np.lexsort((d3[:, 0],))
    d3a = d3[d3ord]
    x = d3a[:, 0]
    d3a[:, 0] = (x - np.min(x)) / (np.max(x) - np.min(x))  # 归一化
    gaprange = sum(np.diff(d3a[:, 0])) / (len(d3a[:, 0]) - 1)  # 计算gap大小

    movetrack = np.repeat(0, 5)
    movetry = np.repeat(0, 5)  # 作者是希望统计对应score变化模式的次数，包括shift up/down, expand/contract, insert/remove, dogleg
    print("\n----------Initial Penalty Calculation----------")
    print("-----start to compute the biostratigraphic range extension-----")
    print(pen_str['biostrat'] + data_offset)

    if pen_str['n_biostrat'] > 0:
        cv = NetRangeExtension(d3a[:, pen_str['biostrat'] + data_offset - 1])
    else:
        cv = 0
    print("biostrat error was: ", cv, "\n")

    if pen_str['n_pmag'] == 1:
        cv += ftransitions2(d3a[:, pen_str['pmag'] + data_offset - 1]) * 30
    else:
        if pen_str['n_pmag'] > 1:
            pass  # ???

    if pen_str['n_ashes'] == 1:
        pass  # ???
    else:
        if pen_str['n_ashes'] > 1:
            pass  # ???

    if pen_str['n_continuous'] == 1:
        pass
    else:
        if pen_str['n_continuous'] > 1:
            pass

    if pen_str['n_dates'] > 0:
        for i in range(pen_str['n_dates']):
            pass

    bestcv = cv
    print("----------Starting penalty----------\n")
    print("current penalty: ", bestcv)
    bestd3 = d3a
    nsections = np.max(d3a[:, 1])
    history = np.mat(np.zeros((nouter, 3)))
    nhorizons = d3a.shape[0]

    for i in range(nouter):
        for j in range(ninner):
            pd3 = d3a
            psec = math.floor(random.uniform(1.000001, nsections + 0.99999))
            pmove = random.random()
            if pmove < 0.2:  # 向上/下移动
                shval = random.uniform(-0.1, 0.1)
                ps = pd3[:, 1] == psec
                pd3[ps, 0] += shval
                movetype = 1
                movetry[0] += 1
            elif pmove < 0.4:  # 扩张/收缩
                shval = random.uniform(-0.05, 0.05) + 1
                ps = pd3[:, 1] == psec
                pd3min = np.mean(pd3[ps, 0])
                pd3[ps, 0] = (pd3[ps, 0] - pd3min) * shval + pd3min
                movetype = 2
                movetry[1] += 1
            elif pmove < 0.6:  # 插入/移除gap
                ps = pd3[:, 1] == psec
                while sum(ps) < 3:  # 这个极端情况下可能会出bug（如果数据集中每个剖面都少于3个）
                    psec = math.floor(random.uniform(1.000001, nsections + 0.99999))
                    ps = pd3[:, 1] == psec
                breakpoint = math.floor(random.uniform(2, sum(ps) - 0.001))
                w = np.where(ps)[0]
                gap = random.uniform(-0.59, 5) * (pd3[w[breakpoint], 0] - pd3[w[breakpoint - 1], 0])
                w = np.delete(w, np.arange(0, breakpoint))
                pd3[w, 0] += gap
                movetype = 3
                movetry[2] += 1
            elif pmove < 0.8:  # 插入dogleg
                shval = random.uniform(-0.1, 0.1) + 1
                ps = pd3[:, 1] == psec
                while sum(ps) < 3:
                    psec = math.floor(random.uniform(1.000001, nsections + 0.99999))
                    ps = pd3[:, 1] == psec
                w = np.where(ps)[0]
                breakpt = math.floor(random.uniform(2, sum(ps) - 0.001))
                gapval = np.diff(pd3[w, 0])
                upchoice = random.uniform(0, 1)
                if upchoice > 0.5:
                    gapval[breakpt:sum(ps) + 1] *= shval
                else:
                    gapval[0:breakpt] *= shval
                newval = np.cumsum(np.append(pd3[w[0], 0], gapval))
                pd3[w, 0] = newval
                movetype = 4
                movetry[3] += 1
            else:  # 插入乱序程序
                target = math.floor(random.uniform(1.000001, nhorizons + 0.99))
                dmove = math.floor(random.uniform(0.01, 1.99))
                nmove = math.ceil(abs(np.random.normal(0, 4, 1)))
                nmove = 1
                movetype = 5
                movetry[4] += 1
                if dmove == 0:
                    startsection = pd3[target - 1, 1]
                    while nmove > 0:
                        if target == nhorizons:
                            nmove = 0
                        else:
                            if pd3[target, 1] == startsection:
                                nmove = 0
                            else:
                                temp = pd3[target, 0]
                                pd3[target, 0] = pd3[target - 1, 0]
                                pd3[target - 1, 0] = temp
                                target += 1
                                nmove -= 1
                else:
                    while nmove > 0:
                        if target == 1:
                            nmove = 0
                        else:
                            if pd3[target - 2, 1] == pd3[target - 1, 1]:
                                target -= 1
                            else:
                                temp = pd3[target - 2, 0]
                                pd3[target - 2, 0] = pd3[target - 1, 0]
                                pd3[target - 1, 0] = temp
                                target -= 1
                                nmove -= 1

            # 对当前方案进行排序
            pd3ord = np.lexsort((pd3[:, 0],))
            pd3 = pd3[pd3ord]

            """
            find the error for the proposed solution, starting with biostrat data


            """
            if pen_str['n_biostrat'] > 0:
                pcv3 = NetRangeExtension(pd3[:, data_offset + pen_str["biostrat"] - 1])
            else:
                pcv3 = 0

            if pcv3 <= bestcv:
                if pcv3 < bestcv:
                    print("current best penalty:", bestcv)
                    movetrack[movetype - 1] += 1
                else:
                    # print("swop\t")
                    pass
                bestcv = pcv3
                bestd3 = pd3
                d3a = pd3
                cv = pcv3
            else:
                pch = random.uniform(0, 1)
                if pch < math.exp(-(pcv3 - cv) / temperature):
                    d3a = pd3
                    cv = pcv3
                else:
                    # don't accept the change
                    pass

            d3a[:, 0] = np.arange(0, len(d3a[:, 0])) / len(d3a[:, 0])

        temperature *= cooling
        history[i, 0] = temperature
        history[i, 1] = bestcv
        history[i, 2] = pcv3
        print("N outer: {:d} | T: {:.5e} | Best pen: {:d} | pick_move: {:.5f} | pick_section: {:d} | "
              "Recent prop pen: {:d} \n".format(i, temperature, bestcv, pmove, psec, pcv3))

    ptime.stop()
    y = {
        "pen": bestcv,
        "initpen": cv,
        "d": bestd3,
        "history": history
    }
    print("Best Total Penalty:", bestcv)

    movetrack = 1000 * movetrack / movetry
    print("Move track values: {:f} {:f} {:f} {:f} {:f}".format(movetrack[0], movetrack[1], movetrack[2],
                                                               movetrack[3], movetrack[4]))
    print("gaprange:", gaprange)

    return pd3