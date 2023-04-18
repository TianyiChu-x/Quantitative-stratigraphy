#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <cmath>


struct Horizon {
    std::string section_name;
    double horizon_score;
    int section_number;
    int horizon_number;
    int horizon_height;
    std::vector<int> presence_absence_data;
};

struct PenaltyParameters {
    int n_biostrat;
    std::vector<int> biostrat_columns;

    int n_pmag;
    int pmag;

    int n_dates;
    std::vector<std::vector<int>> dates;

    int n_ashes;
    std::vector<std::vector<int>> ashes;

    int n_continuous;
    std::vector<std::vector<int>> continuous;
};

std::vector<Horizon> read_csv_data(const std::string& file_path) {
    std::vector<Horizon> horizons;
    std::ifstream file(file_path);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open the file." << std::endl;
        return horizons;
    }

    // Ignore the header line
    std::string header_line;
    std::getline(file, header_line);

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        Horizon horizon;

        std::getline(ss, token, ',');
        horizon.section_name = token;

        std::getline(ss, token, ',');
        horizon.horizon_score = std::stod(token);

        std::getline(ss, token, ',');
        horizon.section_number = std::stoi(token);

        std::getline(ss, token, ',');
        horizon.horizon_number = std::stoi(token);

        std::getline(ss, token, ',');
        horizon.horizon_height = std::stoi(token);

        while (std::getline(ss, token, ',')) {
            horizon.presence_absence_data.push_back(std::stoi(token));
        }

        horizons.push_back(horizon);
    }

    file.close();
    return horizons;
}

PenaltyParameters initialize_penalty_parameters() {
    PenaltyParameters params;

//    params.n_biostrat = 62;
    params.n_biostrat = 4522;
    for (int i = 1; i <= 62; ++i) {
        params.biostrat_columns.push_back(i);
    }

    params.n_pmag = 0;
    params.pmag = 63;

    params.n_dates = 0;
    params.dates = {
        {109, 2, 110, 1, 100},
        {111, 2, 112, 1, 100},
        {113, 2, 114, 1, 100}
    };

    params.n_ashes = 0;
    params.ashes = {
        {68, 100},
        {69, 100}
    };

    params.n_continuous = 0;
    params.continuous = {
        {70, 5},
        {71, 5}
    };

    return params;
}

int calculate_penalty(const std::vector<Horizon>& horizons, int n_biostrat) {
    int penalty = 0;

    for (int column = 0; column < n_biostrat; ++column) {
        bool first_one_found = false;
        int first_one_index = 0;
        int last_one_index = 0;

        for (int row = 0; row < horizons.size(); ++row) {
            int value = horizons[row].presence_absence_data[column];
            if (value == 1) {
                if (!first_one_found) {
                    first_one_found = true;
                    first_one_index = row;
                }
                last_one_index = row;
            }
        }

        for (int row = first_one_index; row <= last_one_index; ++row) {
            if (horizons[row].presence_absence_data[column] == 0) {
                ++penalty;
            }
        }
    }

    return penalty;
}


void HorizonAnneal(std::vector<Horizon>& horizons, const PenaltyParameters& penalty_parameters, int nouter, int ninner, double temperature, double cooling) {
    // Sort horizons by horizon_score
    std::sort(horizons.begin(), horizons.end(), [](const Horizon& a, const Horizon& b) { return a.horizon_score < b.horizon_score; });

    // Calculate min and max horizon_score
    double min_horizon_score = horizons.front().horizon_score;
    double max_horizon_score = horizons.back().horizon_score;

    // Normalize horizon_score for each horizon
    for (Horizon& horizon : horizons) {
        horizon.horizon_score = (horizon.horizon_score - min_horizon_score) / (max_horizon_score - min_horizon_score);
    }
    
    int initial_penalty = 0;
    int best_penalty = 0;
    if (penalty_parameters.n_biostrat > 0) {
        initial_penalty = calculate_penalty(horizons, penalty_parameters.n_biostrat);
    }
    std::cout << "Initial penalty: " << initial_penalty << std::endl;
    best_penalty = initial_penalty;
    int num_sections = 0;
    int num_horizons = horizons.size();
    for (const Horizon& horizon : horizons) {
        if (horizon.section_number > num_sections) {
            num_sections = horizon.section_number;
        }
    }
    
    // 初始化随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::uniform_int_distribution<> dis_int(1, num_sections);
    std::uniform_real_distribution<> dis_move(-0.1, 0.1);
    std::uniform_real_distribution<> dis_scale(-0.05, 0.05);
    std::uniform_real_distribution<> dis_gap(-0.59, 5.0);
    std::vector<Horizon> d3 = horizons;

    for (int i = 0; i < nouter; ++i) {
        for (int j = 0; j < ninner; ++j) {
            std::vector<Horizon> pd3 = d3;
            double pmove = dis(gen);
            int psec = dis_int(gen);
            // 向上/向下移动
            if (pmove < 0.2) {
                double random_move = dis_move(gen);
                for (Horizon& horizon : pd3) {
                    if (horizon.section_number == psec) {
                        horizon.horizon_score += random_move;
                    }
                }
            } else if (pmove < 0.4) {
                // 扩张/收缩
                double random_scale = dis_scale(gen) + 1.0;
                double sum = 0.0;
                int count = 0;
                for (const Horizon& horizon : pd3) {
                    if (horizon.section_number == psec) {
                        sum += horizon.horizon_score;
                        count++;
                    }
                }
                double avg = sum / count;
                for (Horizon& horizon : pd3) {
                    if (horizon.section_number == psec) {
                        horizon.horizon_score = (horizon.horizon_score - avg) * random_scale + avg;
                    }
                }
            } else if (pmove < 0.6) {
                // 插入/移除gap
                int count = 0;
                for (const Horizon& horizon : pd3) {
                    if (horizon.section_number == psec) {
                        count++;
                    }
                }
                while (count < 3) {
                    psec = dis_int(gen);
                    count = 0;
                    for (const Horizon& horizon : pd3) {
                        if (horizon.section_number == psec) {
                            count++;
                        }
                    }
                }
                std::uniform_int_distribution<> dis_breakpoint(2, count - 1);
                int breakpoint = dis_breakpoint(gen);

                double gap = 0.0;
                int current_count = 0;
                for (int k = 0; k < pd3.size(); ++k) {
                    if (pd3[k].section_number == psec) {
                        current_count++;
                        if (current_count == breakpoint + 1) {
                            gap = (pd3[k].horizon_score - pd3[k - 1].horizon_score) * dis_gap(gen);
                            break;
                        }
                    }
                }
                current_count = 0;
                for (Horizon& horizon : pd3) {
                    if (horizon.section_number == psec) {
                        current_count++;
                        if (current_count <= breakpoint + 1) {
                            horizon.horizon_score += gap;
                        }
                    }
                }
            } else if (pmove < 0.8) {
                // dogleg
                double shval = dis_move(gen) + 1.0;
                std::vector<bool> ps(pd3.size());
                for (size_t i = 0; i < pd3.size(); ++i) {
                    ps[i] = pd3[i].section_number == psec;
                }
                while (std::count(ps.begin(), ps.end(), true) < 3) {
                    psec = dis_int(gen);
                    for (size_t i = 0; i < pd3.size(); ++i) {
                        ps[i] = pd3[i].section_number == psec;
                    }
                }
                std::vector<size_t> w;
                for (size_t i = 0; i < ps.size(); ++i) {
                    if (ps[i]) {
                        w.push_back(i);
                    }
                }
                std::uniform_int_distribution<> dis_breakpt(2, static_cast<double>(w.size()) - 1);
                int breakpt = dis_breakpt(gen);
                std::vector<double> gapval(w.size() - 1);
                for (size_t i = 0; i < gapval.size(); ++i) {
                    gapval[i] = pd3[w[i + 1]].horizon_score - pd3[w[i]].horizon_score;
                }
                double upchoice = dis(gen);
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
                newval[0] = pd3[w[0]].horizon_score;
                for (size_t i = 1; i < newval.size(); ++i) {
                    newval[i] = newval[i - 1] + gapval[i - 1];
                }
                for (size_t i = 0; i < w.size(); ++i) {
                    pd3[w[i]].horizon_score = newval[i];
                }
            } else {
                // 插入乱序程序的策略
                std::uniform_int_distribution<> dis_target(1, num_horizons);
                int target = dis_target(gen);
                std::uniform_int_distribution<> dis_dmove(0, 1);
                int dmove = dis_dmove(gen);
                int nmove = std::ceil(std::abs(std::normal_distribution<double>(0, 4)(gen)));
                nmove = 1;

                if (dmove == 0) {
                    int startsection = pd3[target - 1].section_number;
                    while (nmove > 0) {
                        if (target == num_horizons) {
                            nmove = 0;
                        } else {
                            if (pd3[target].section_number == startsection) {
                                nmove = 0;
                            } else {
                                double temp = pd3[target].horizon_score;
                                pd3[target].horizon_score = pd3[target - 1].horizon_score;
                                pd3[target - 1].horizon_score = temp;
                                target += 1;
                                nmove -= 1;
                            }
                        }
                    }
                } else {
                    while (nmove > 0) {
                        if (target == 1) {
                            nmove = 0;
                        } else {
                            if (pd3[target - 2].section_number == pd3[target - 1].section_number) {
                                target -= 1;
                            } else {
                                double temp = pd3[target - 2].horizon_score;
                                pd3[target - 2].horizon_score = pd3[target - 1].horizon_score;
                                pd3[target - 1].horizon_score = temp;
                                target -= 1;
                                nmove -= 1;
                            }
                        }
                    }
                }
            }
            // Sort horizons by horizon_score
            std::sort(pd3.begin(), pd3.end(), [](const Horizon& a, const Horizon& b) { return a.horizon_score < b.horizon_score; });
            int current_penalty = 0;
            if (penalty_parameters.n_biostrat > 0) {
                current_penalty = calculate_penalty(pd3, penalty_parameters.n_biostrat);
            }
            if (current_penalty <= best_penalty) {
                if (current_penalty < best_penalty) {
                    // std::cout << "best penalty: " << current_penalty << std::endl;
                }
                d3 = pd3;
                best_penalty = current_penalty;
                initial_penalty = current_penalty;
            } else {
                double pch = std::uniform_real_distribution<double>{ 0.0, 1.0 }(gen);
                if (pch < exp(-(current_penalty - best_penalty) / temperature)) {
                    d3 = pd3;
                    initial_penalty = current_penalty;
                } else {
                    pd3 = d3;
                }
            }
            size_t numRows = horizons.size();
            for (size_t i = 0; i < numRows; ++i) {
                horizons[i].horizon_score = static_cast<double>(i) / static_cast<double>(numRows);
            }

        }
        temperature *= cooling;
        std::cout << "N outer: " << i << " | T: " << temperature << " | Best pen: " << best_penalty << " \n";
    }
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    
    const std::string file_path = "/Users/qianxian/PycharmProjects/HA/dataset/Horizon_Data_sigmoid_for_HA.csv";
    std::vector<Horizon> horizons = read_csv_data(file_path);
    PenaltyParameters penalty_parameters = initialize_penalty_parameters();
    
    int nouter = 400;
    int ninner = 100;
    double temperature = 5.0;
    double cooling = 0.50;
    HorizonAnneal(horizons, penalty_parameters, nouter, ninner, temperature, cooling);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "\nElapsed time: " << elapsed.count() << " seconds." << std::endl;
    return 0;
}
