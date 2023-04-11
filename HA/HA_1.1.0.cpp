#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <limits>
#include <random>
#include <cmath>

using namespace std;

vector<int> produce_list() {
    constexpr int n_biostrat = 4522;
    constexpr int pmag = 63;

    vector<int> biostrat(n_biostrat);
    std::iota(biostrat.begin(), biostrat.end(), 1);

    int n_pmag = 0;
    int n_dates = 0;

    vector<vector<int>> dates{{109, 2, 110, 1, 100}, {111, 2, 112, 1, 100}, {113, 2, 114, 1, 100}};
    int n_ashes = 0;
    vector<vector<int>> ashes{{68, 100}, {69, 100}};
    int n_continuous = 0;
    vector<vector<int>> continuous{{70, 5}, {71, 5}};

    vector<int> penalty_spec_62;
    penalty_spec_62.reserve(n_biostrat + 4 + (dates.size() * 5) + 1 + (ashes.size() * 2) + 1 + (continuous.size() * 2));

    penalty_spec_62.emplace_back(n_biostrat);
    penalty_spec_62.insert(penalty_spec_62.end(), biostrat.begin(), biostrat.end());
    penalty_spec_62.emplace_back(n_pmag);
    penalty_spec_62.emplace_back(pmag);
    penalty_spec_62.emplace_back(n_dates);
    for (const auto& row : dates) {
        penalty_spec_62.insert(penalty_spec_62.end(), row.begin(), row.end());
    }
    penalty_spec_62.emplace_back(n_ashes);
    for (const auto& row : ashes) {
        penalty_spec_62.insert(penalty_spec_62.end(), row.begin(), row.end());
    }
    penalty_spec_62.emplace_back(n_continuous);
    for (const auto& row : continuous) {
        penalty_spec_62.insert(penalty_spec_62.end(), row.begin(), row.end());
    }

    return penalty_spec_62;
}

bool compare_first_column(const std::vector<double>& a, const std::vector<double>& b) {
    return a[0] < b[0];
}


int NetRangeExtension(const std::vector<std::vector<double>>& data) {
    int total_sum = 0;

    for (size_t j = 0; j < data[0].size(); ++j) { // 对每一列进行计算
        int lp_begin = -1, lp_end = -1;
        for (size_t i = 0; i < data.size(); ++i) {
            if (lp_begin == -1 && data[i][j] == 1) {
                lp_begin = i;
            }
            if (data[i][j] == 1) {
                lp_end = i;
            }
        }
        int column_sum = 0;
        for (int i = lp_begin; i <= lp_end; ++i) {
            if (data[i][j] == 0) {
                column_sum++;
            }
        }
        total_sum += column_sum;
    }
    return total_sum;
}

void HorizonAnneal(std::vector<std::vector<string>>& dataframe, int n_biostrat, vector<int> biostrat, int nouter = 400, int ninner = 100, double temperature = 5, double cooling = 0.9) {
    const int data_offset = 4;
    std::vector<std::string> columns = dataframe[0];
    std::vector<std::vector<double>> d3(dataframe.size() - 1, std::vector<double>(dataframe[0].size()));

    for (int i = 1; i < dataframe.size(); ++i) {
        for (int j = 0; j < dataframe[i].size(); ++j) {
            d3[i - 1][j] = std::stod(dataframe[i][j]);
        }
    }

    std::sort(d3.begin(), d3.end(), compare_first_column);

    double min_val = d3[0][0];
    double max_val = d3.back()[0];
    for (auto &row : d3) {
        row[0] = (row[0] - min_val) / (max_val - min_val);
    }

    std::vector<double> movetrack(5, 0);
    std::vector<double> movetry(5, 0);

    double cv;
    if (n_biostrat > 0) {
        int col_index = biostrat[0] + data_offset - 1;
        std::vector<std::vector<double>> d3_col(d3.size(), std::vector<double>(d3[0].size() - col_index));
        for (int i = 0; i < d3.size(); ++i) {
            std::copy(d3[i].begin() + col_index, d3[i].end(), d3_col[i].begin());
        }
        cv = NetRangeExtension(d3_col);
    } else {
        cv = 0;
    }

    double bestcv = cv;
    std::cout << "----------Starting penalty----------\n" << std::endl;
    std::cout << "current penalty: " << bestcv << std::endl;
    std::vector<std::vector<double>> bestd3 = d3;
    size_t nsections = (*std::max_element(d3.cbegin(), d3.cend(), [](const std::vector<double>& a, const std::vector<double>& b) { return a[1] < b[1]; }))[1];
    size_t nhorizons = d3.size();

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<double> dist_sec(1.000001, nsections + 0.99999);
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    for (int i = 0; i < nouter; ++i) {
        for (int j = 0; j < ninner; ++j) {
            int movetype;
            std::vector<std::vector<double>> pd3 = d3;
            double psec = static_cast<int>(std::floor(dist_sec(rng)));
            double pmove = dis(rng);

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
                double shval = std::uniform_real_distribution<double>{ -0.05, 0.05 }(rng)+1;
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
                    for (size_t i = breakpt; i < w.size() - 1; ++i) {
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

            std::sort(pd3.begin(), pd3.end(),compare_first_column);
            if (n_biostrat > 0) {
                int col_index = biostrat[0] + data_offset - 1;
                std::vector<std::vector<double>> pd3_col(pd3.size(), std::vector<double>(pd3[0].size() - col_index));
                for (int k = 0; k < pd3.size(); ++k) {
                    std::copy(pd3[k].begin() + col_index, pd3[k].end(), pd3_col[k].begin());
                }
                cv = NetRangeExtension(pd3_col);
            } else {
                cv = 0;
            }

            double delta = cv - bestcv;
            if (delta < 0 || dis(rng) < std::exp(-delta / temperature)) {
                d3 = pd3;
                bestcv = cv;
                movetrack[movetype] += 1;
            } else {
                movetry[movetype] += 1;
            }
        }
        temperature *= cooling;
        std::cout << "N outer: " << i << " | T: " << temperature << " | Best pen: " << bestcv << " \n";
    }

    std::vector<std::vector<std::string>> result(d3.size() + 1, std::vector<std::string>(columns.size()));
    result[0] = columns;
    for (int i = 0; i < d3.size(); ++i) {
        for (int j = 0; j < d3[i].size(); ++j) {
            result[i + 1][j] = std::to_string(d3[i][j]);
        }
    }

    dataframe = result;
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    string filename = "/Users/qianxian/PycharmProjects/HA/dataset/Horizon_Data_sigmoid_for_HA.csv";
//    string filename = "Horizon_Data_sigmoid_for_HA.csv";
    ifstream file(filename, ios::binary);

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
        getline(ss, cell, ',');
        while (ss.good()) {
            getline(ss, cell, ',');
            row.push_back(cell);
        }

        data.push_back(row);
    }

    file.close();
    vector<int> penalty_spec_62 = produce_list();
    int n_biostrat = penalty_spec_62[0];
    vector<int> biostrat(n_biostrat);
    for (int i = 0; i < n_biostrat; i++) {
        biostrat[i] = penalty_spec_62[i + 1];
    }
    HorizonAnneal(data, n_biostrat, biostrat);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "\nElapsed time: " << elapsed.count() << " seconds." << std::endl;
    return 0;
}
