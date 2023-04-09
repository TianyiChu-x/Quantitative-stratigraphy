#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <random> // std::uniform_real_distribution, std::default_random_engine
#include <cmath>  // std::floor

using namespace std;

vector<int> produce_list() {
    //int n_biostrat = 62;
    int n_biostrat = 4522;
    vector<int> biostrat(n_biostrat);
    for (int i = 0; i < n_biostrat; i++) {
        biostrat[i] = i + 1;
    }

    int n_pmag = 0;
    int pmag = 63;

    int n_dates = 0;
    vector<vector<int>> dates{ {109, 2, 110, 1, 100}, {111, 2, 112, 1, 100}, {113, 2, 114, 1, 100} };

    int n_ashes = 0;
    vector<vector<int>> ashes{ {68, 100}, {69, 100} };

    int n_continuous = 0;
    vector<vector<int>> continuous{ {70, 5}, {71, 5} };

    vector<int> penalty_spec_62;
    penalty_spec_62.push_back(n_biostrat);
    for (const auto& element : biostrat) {
        penalty_spec_62.push_back(element);
    }
    penalty_spec_62.push_back(n_pmag);
    penalty_spec_62.push_back(pmag);
    penalty_spec_62.push_back(n_dates);
    for (const auto& row : dates) {
        for (const auto& element : row) {
            penalty_spec_62.push_back(element);
        }
    }
    penalty_spec_62.push_back(n_ashes);
    for (const auto& row : ashes) {
        for (const auto& element : row) {
            penalty_spec_62.push_back(element);
        }
    }
    penalty_spec_62.push_back(n_continuous);
    for (const auto& row : continuous) {
        for (const auto& element : row) {
            penalty_spec_62.push_back(element);
        }
    }

    return penalty_spec_62;
}

bool compare_first_column(const std::vector<double>& a, const std::vector<double>& b) {
    return a[0] < b[0]; // 按照第一列升序排列
}

int ColumnRangeExtension(std::vector<double>& y) {
    int lp_begin, lp_end;
    for (int i = 0; i < y.size(); i++) {
        if (y[i] == 1) {
            lp_begin = i;
            break;
        }
    }
    for (int i = y.size() - 1; i >= 0; i--) {
        if (y[i] == 1) {
            lp_end = i;
            break;
        }
    }
    int lpv_size = lp_end - lp_begin + 1;
    std::vector<double> lpv(lpv_size);
    for (int i = 0; i < lpv_size; i++) {
        lpv[i] = lp_begin + i;
    }
    int sum = 0;
    for (int i = 0; i < lpv_size; i++) {
        if (y[lpv[i]] == 0) {
            sum++;
        }
    }
    return sum;
}

int NetRangeExtension(const std::vector<std::vector<double>>& data) {
    std::vector<double> result(data[0].size()); // 定义一个保存结果的一维 vector，长度等于 data 的列数
    for (size_t j = 0; j < data[0].size(); ++j) { // 对每一列进行计算
        std::vector<double> column(data.size()); // 取出当前列的数据
        for (size_t i = 0; i < data.size(); ++i) {
            column[i] = data[i][j];
        }
        result[j] = ColumnRangeExtension(column); // 计算结果
    }
    return std::accumulate(result.begin(), result.end(), 0.0); // 对result中的每个元素求和
}

void HorizonAnneal(std::vector<std::vector<string>>& dataframe,int n_biostrat, vector<int> biostrat, int nouter = 400, int ninner = 100, double temperature = 5, double cooling = 0.9) {
	int pcv3 = 0, data_offset = 4;
	double pmove = 0, psec = 0, cv;
    std::vector<std::string> columns;  // 存储表头，数据类型为字符串
    std::vector<std::vector<std::string>> d3_string(dataframe.begin() + 1, dataframe.end());
    std::vector<std::vector<double>> d3(d3_string.size());
    for (int i = 0; i < d3_string.size(); ++i) {
        for (int j = 0; j < d3_string[i].size(); ++j) {
            d3[i].push_back(std::stod(d3_string[i][j]));
        }
    }
	for (int i = 0; i < dataframe[0].size(); ++i) {
		columns.push_back(dataframe[0][i]);  // 将每列的表头添加到 vector 中
	}
    std::sort(d3.begin(), d3.end(), compare_first_column);  // 根据score对数据排序
    // 获取score的最大值和最小值
    double min_val = d3[0][0];
    double max_val = d3[d3.size() - 1][0];
    // 归一化score
    for (vector<double>& row : d3) {
        row[0] = (row[0] - min_val) / (max_val - min_val);
    }
    vector<double> movetrack(5, 0);
    vector<double> movetry(5, 0);
    cout << "\n----------Initial Penalty Calculation----------" << endl;
    cout << "-----start to compute the biostratigraphic range extension-----" << endl;
    if (n_biostrat > 0) {
        // 获取d3a中指定列的数据并传入NetRangeExtension函数进行计算
        int col_index = biostrat[0] + data_offset - 1;
        vector<vector<double>> d3_col;
        for (auto row : d3) {
            std::vector<double> new_row(row.begin() + col_index, row.end());
            d3_col.push_back(new_row);
        }
        cv = NetRangeExtension(d3_col);
    }
    else {
        cv = 0;
    }
    double bestcv = cv;
    std::cout << "----------Starting penalty----------\n" << std::endl;
    std::cout << "current penalty: " << bestcv << std::endl;
    std::vector<std::vector<double>> bestd3 = d3;
    size_t nsections = (*std::max_element(d3.cbegin(), d3.cend(), [](const std::vector<double>& a, const std::vector<double>& b) { return a[1] < b[1]; }))[1];
    size_t nhorizons = d3.size();
    cout << nhorizons << endl;


    // 输出排序后的结果
    //for (const auto& row : col_data) {
    //    for (const auto& value : row) {
    //        std::cout << value << " ";
    //    }
    //    std::cout << std::endl;
    //}

	//for (const auto& element : d3) {
	//	cout << element << " ";
	//}
    // 
	//for (const auto& row : d3) {
	//	for (const auto& cell : row) {
	//		cout << cell << " ";
	//	}
	//	cout << endl;
	//}

    // 模拟退火算法
    for (int i = 0; i < nouter; ++i) {
        for (int j = 0; j < ninner; ++j) {
            int movetype = 0;
            std::vector<std::vector<double>> pd3 = d3;
            std::default_random_engine rng(std::random_device{}()); // 初始化一个随机数生成器
            std::uniform_real_distribution<double> dist_sec(1.000001, nsections + 0.99999); // 初始化一个均匀分布
            psec = static_cast<int>(std::floor(dist_sec(rng))); // 产生一个整型随机数 psec，值为区间 [1, nsections] 内的一个整数（向下取整）
            // 生成（0，1）的随机数pmove
            std::mt19937 gen(std::random_device{}());
            std::uniform_real_distribution<> dis(0.0, 1.0);
            pmove = dis(gen);
            if (pmove < 0.2) { // 向上/下移动
                double shval = std::uniform_real_distribution<double>{ -0.1, 0.1 }(rng);
                for (auto& row : pd3) {
                    if (row[1] == psec) {
                        row[0] += shval;
                    }
                }
                movetype = 1;
                ++movetry[0];
            } 
            else if (pmove < 0.4) {  // 扩张/收缩
                double shval = std::uniform_real_distribution<double>{ -0.05, 0.05 }(rng) + 1;
                double score_sum = 0;
                int count = 0;
                for (auto& row : pd3) {
                    if (row[1] == psec) {
                        count++;
                        score_sum += row[0];
                    }
                }
                double pd3min = score_sum / count;
                for (auto& row : pd3) {
                    if (row[1] == psec) {
                        row[0] = (row[0] - pd3min) * shval + pd3min;
                    }
                }
                movetype = 2;
                ++movetry[1];
            } 
            else if (pmove < 0.6) {  // 插入/移除gap
                std::vector<bool> ps(pd3.size(), false);
                for (int i = 0; i < pd3.size(); ++i) {
                    if (pd3[i][1] == psec) {
                        ps[i] = true;
                    }
                }
                while (std::count(ps.begin(), ps.end(), true) < 3) {
                    psec = std::floor(std::uniform_real_distribution<double>(1.000001, nsections + 0.99999)(rng));
                    for (int i = 0; i < pd3.size(); ++i) {
                        if (pd3[i][1] == psec) {
                            ps[i] = true;
                        }
                    }
                }
                int sum_ps = std::count(ps.begin(), ps.end(), true);
                int breakpoint = std::floor(std::uniform_real_distribution<double>(2, sum_ps - 0.001)(rng));
                std::vector<int> w;
                for (int i = 0; i < ps.size(); ++i) {
                    if (ps[i]) {
                        w.push_back(i);
                    }
                }
                double pd3_breakpoint_prev = pd3[w[breakpoint - 1]][0];
                double pd3_breakpoint = pd3[w[breakpoint]][0];
                double gap = std::uniform_real_distribution<double>(-0.59, 5)(rng) * (pd3_breakpoint - pd3_breakpoint_prev);
                for (int i = 0; i < w.size(); ++i) {
                    pd3[w[i]][0] += gap;
                }
                movetype = 3;
                ++movetry[2];
            } 
            else if (pmove < 0.8) {
                double shval = std::uniform_real_distribution<double>{ -0.1, 0.1 }(rng)+1;
                std::vector<bool> ps(pd3.size());
                for (size_t i = 0; i < pd3.size(); ++i) {
                    ps[i] = pd3[i][1] == psec;
                }
                while (std::count(ps.begin(), ps.end(), true) < 3) {
                    psec = std::floor(std::uniform_real_distribution<double>{1.000001, nsections + 0.99999}(rng));
                    for (size_t i = 0; i < pd3.size(); ++i) {
                        ps[i] = pd3[i][1] == psec;
                    }
                }
                std::vector<size_t> w;
                for (size_t i = 0; i < ps.size(); ++i) {
                    if (ps[i]) {
                        w.push_back(i);
                    }
                }
                size_t breakpt = std::floor(std::uniform_real_distribution<double>{2, static_cast<double>(w.size()) - 0.001}(rng));
                std::vector<double> gapval(w.size() - 1);
                for (size_t i = 0; i < gapval.size(); ++i) {
                    gapval[i] = pd3[w[i + 1]][0] - pd3[w[i]][0];
                }
                double upchoice = std::uniform_real_distribution<double>{ 0, 1 }(rng);
                if (upchoice > 0.5) {
                    for (size_t i = breakpt; i < w.size()-1; ++i) {
                        gapval[i] *= shval;
                    }
                }
                else {
                    for (size_t i = 0; i < breakpt; ++i) {
                        gapval[i] *= shval;
                    }
                }
                std::vector<double> newval(w.size());
                newval[0] = pd3[w[0]][0];
                for (size_t i = 1; i < newval.size(); ++i) {
                    newval[i] = newval[i - 1] + gapval[i - 1];
                }
                for (size_t i = 0; i < w.size(); ++i) {
                    pd3[w[i]][0] = newval[i];
                }
                movetype = 4;
                ++movetry[3];
            } 
            else {
                double target = std::floor(std::uniform_real_distribution<double>(1.000001, nhorizons + 0.99)(rng));
                int dmove = std::floor(std::uniform_real_distribution<double>(0.01, 1.99)(rng));
                int nmove = std::ceil(std::abs(std::normal_distribution<double>(0, 4)(rng)));
                nmove = 1;
                int movetype = 5;
                movetry[4] += 1;
                if (dmove == 0) {
                    double startsection = pd3[target - 1][1];
                    while (nmove > 0) {
                        if (target == nhorizons) {
                            nmove = 0;
                        }
                        else {
                            if (pd3[target][1] == startsection) {
                                nmove = 0;
                            }
                            else {
                                double temp = pd3[target][0];
                                pd3[target][0] = pd3[target - 1][0];
                                pd3[target - 1][0] = temp;
                                target += 1;
                                nmove -= 1;
                            }
                        }
                    }
                }
                else {
                    while (nmove > 0) {
                        if (target == 1) {
                            nmove = 0;
                        }
                        else {
                            if (pd3[target - 2][1] == pd3[target - 1][1]) {
                                target -= 1;
                            }
                            else {
                                double temp = pd3[target - 2][0];
                                pd3[target - 2][0] = pd3[target - 1][0];
                                pd3[target - 1][0] = temp;
                                target -= 1;
                                nmove -= 1;
                            }
                        }
                    }
                }
            }
            std::sort(pd3.begin(), pd3.end(), compare_first_column);  // 根据score对数据排序
            if (n_biostrat > 0) {
                // 获取pd3中指定列的数据并传入NetRangeExtension函数进行计算
                int col_index = biostrat[0] + data_offset - 1;
                vector<vector<double>> pd3_col;
                for (auto row : pd3) {
                    std::vector<double> new_row(row.begin() + col_index, row.end());
                    pd3_col.push_back(new_row);
                }
                pcv3 = NetRangeExtension(pd3_col);
            }
            else {
                pcv3 = 0;
            }

            if (pcv3 <= bestcv) {
                if (pcv3 < bestcv) {
                    std::cout << "best penalty: " << bestcv << std::endl;
                    movetrack[movetype] += 1;
                }
                bestcv = pcv3;
                bestd3 = pd3;
                d3 = pd3;
                cv = pcv3;
            }
            else {
                double pch = std::uniform_real_distribution<double>{ 0.0, 1.0 }(rng);
                if (pch < exp(-(pcv3 - cv) / temperature)) {
                    d3 = pd3;
                    cv = pcv3;
                }
            }
            for (int i = 0; i < d3.size(); i++) {
                d3[i][0] = i / double(d3.size());
            }

            // cout << shval << " ";

            
        }

        temperature *= cooling;
        std::cout << "N outer: " << i << " | T: " << temperature << " | Best pen: " << bestcv << " | pick_move: " << pmove << " | pick_section: " << psec << " | Recent prop pen: " << pcv3 << " \n";
    }
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    //string filename = "riley_62_for_R.csv";
    string filename = "Horizon_Data_sigmoid_for_HA.csv";
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Error opening file\n";
        return 1;
    }

    vector<vector<string>> data;
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        vector<string> row;

        string cell;
        getline(ss, cell, ','); // Skip the first cell in each row
        while (ss.good()) {
            getline(ss, cell, ',');
            row.push_back(cell);
        }

        data.push_back(row);
    }

    file.close();

    //获取输入参数
    vector<int> penalty_spec_62 = produce_list();
    int n_biostrat = penalty_spec_62[0];
    vector<int> biostrat(n_biostrat);
    for (int i = 0; i < n_biostrat; i++){
        biostrat[i] = penalty_spec_62[i + 1];
    }

    HorizonAnneal(data, n_biostrat, biostrat);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "\nElapsed time: " << elapsed.count() << " seconds." << std::endl;

    return 0;
}
