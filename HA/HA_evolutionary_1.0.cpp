#include <algorithm>
#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <limits>
#include <tuple>
#include <fstream>
#include <sstream>
#include <string>
#include <ctime>
#include <chrono>
#include <unordered_set>

struct Horizon {
    std::string section_name;
    double horizon_score;
    int section_number;
    int horizon_number;
    int horizon_height;
    std::vector<int> presence_absence_data;
};

struct Chromosome {
    std::vector<Horizon> horizons;
    double fitness;
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

std::string header_line;
std::vector<Horizon> read_csv_data(const std::string& file_path, std::string &header_line) {
    std::vector<Horizon> horizons;
    std::ifstream file(file_path);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open the file." << std::endl;
        return horizons;
    }

    // Ignore the header line
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

void save_to_csv(const std::vector<Horizon>& horizons, const std::string& file_path) {
    std::ofstream output_file(file_path);

    if (!output_file.is_open()) {
        std::cerr << "Error: Unable to open the output file." << std::endl;
        return;
    }

    // Write header
    output_file << header_line << std::endl;

    // Write data
    for (const auto& horizon : horizons) {
        output_file << horizon.section_name << ","
                    << horizon.horizon_score << ","
                    << horizon.section_number << ","
                    << horizon.horizon_number << ","
                    << horizon.horizon_height;

        for (int data : horizon.presence_absence_data) {
            output_file << "," << data;
        }
        output_file << "\n";
    }
    output_file.close();
    std::cout << "Results saved to: " << file_path << std::endl;
}

PenaltyParameters initialize_penalty_parameters() {
    PenaltyParameters params;

    params.n_biostrat = 62;
//    params.n_biostrat = 4522;
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

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);

Chromosome init_chromosome(const std::vector<Horizon>& d3) {
    Chromosome chromosome;
    chromosome.horizons = d3;
    chromosome.fitness = 0;
    return chromosome;
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

double calculate_fitness(const std::vector<Horizon>& horizons, int n_biostrat) {
    return static_cast<double>(calculate_penalty(horizons, n_biostrat));
}

std::vector<Chromosome> selection(const std::vector<Chromosome>& population, int tournament_size) {
    std::vector<Chromosome> selected_population;
    size_t population_size = population.size();

    // 随机选择器
    std::uniform_int_distribution<size_t> random_selector(0, population_size - 1);

    // 选择新种群
    for (size_t i = 0; i < population_size; ++i) {
        Chromosome best_candidate;
        double best_fitness = std::numeric_limits<double>::infinity();

        // 随机选择 tournament_size 个染色体并找到具有最高适应度的染色体
        for (int j = 0; j < tournament_size; ++j) {
            size_t candidate_index = random_selector(gen);
            const Chromosome& candidate = population[candidate_index];

            if (candidate.fitness < best_fitness) {
                best_fitness = candidate.fitness;
                best_candidate = candidate;
            }
        }

        // 将最佳染色体添加到新种群中
        selected_population.push_back(best_candidate);
    }
//    Chromosome best_candidate;
//    double best_fitness = std::numeric_limits<double>::infinity();
//    for (int i = 0; i < population_size; ++i) {
//        size_t candidate_index = random_selector(gen);
//        const Chromosome& candidate = population[candidate_index];
//
//        if (candidate.fitness < best_fitness) {
//            best_fitness = candidate.fitness;
//            best_candidate = candidate;
//        }
//    }
//    for (int i = 0; i < population_size; ++i) {
//        selected_population.push_back(best_candidate);
//    }
    

    return selected_population;
}

std::tuple<std::vector<Horizon>, std::vector<Horizon>> crossover(const std::vector<Horizon>& parent1, const std::vector<Horizon>& parent2) {
//    std::vector<Horizon> offspring1 = parent1;
//    std::vector<Horizon> offspring2 = parent2;
//
//    int n = parent1.size();
//    std::uniform_int_distribution<int> start_dist(0, n - 1);
//    std::uniform_int_distribution<int> crossover_size_dist(1, n);
//
//    int start = start_dist(gen);
//    int crossover_size = crossover_size_dist(gen);
//
//    int end = (start + crossover_size) % n;
//    if (end > start) {
//        std::vector<Horizon> segment1(parent1.begin() + start, parent1.begin() + end);
//        std::vector<Horizon> segment2(parent2.begin() + start, parent2.begin() + end);
//
//        std::copy(segment2.begin(), segment2.end(), offspring1.begin() + start);
//        std::copy(segment1.begin(), segment1.end(), offspring2.begin() + start);
//    } else {
//        std::vector<Horizon> segment1(parent1.begin() + start, parent1.end());
//        std::vector<Horizon> segment2(parent2.begin() + start, parent2.end());
//
//        segment1.insert(segment1.end(), parent1.begin(), parent1.begin() + end);
//        segment2.insert(segment2.end(), parent2.begin(), parent2.begin() + end);
//
//        std::copy(segment2.begin(), segment2.begin() + parent1.size() - start, offspring1.begin() + start);
//        std::copy(segment2.begin() + parent1.size() - start, segment2.end(), offspring1.begin());
//
//        std::copy(segment1.begin(), segment1.begin() + parent2.size() - start, offspring2.begin() + start);
//        std::copy(segment1.begin() + parent2.size() - start, segment1.end(), offspring2.begin());
//    }
//
//    return std::make_pair(offspring1, offspring2);
    std::vector<Horizon> offspring1 = parent1;
    std::vector<Horizon> offspring2 = parent2;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, int(parent1.size() - 1));
    std::uniform_real_distribution<> dis_move(-1.0/parent1.size(), 1.0/parent1.size());

    std::unordered_set<int> selected_indices;
    int crossover_size = int(1 * parent1.size() / 2);

    while (selected_indices.size() < crossover_size) {
        int index = dis(gen);
        if (selected_indices.find(index) == selected_indices.end()) {
            selected_indices.insert(index);

            int section_number = offspring1[index].section_number;
            int horizon_number = offspring1[index].horizon_number;

            // Find the corresponding horizon in offspring2
            auto it = std::find_if(offspring2.begin(), offspring2.end(), [section_number,horizon_number](const Horizon& horizon) { return horizon.section_number == section_number & horizon.horizon_number == horizon_number; });

            // Swap horizon_scores
            if (it != offspring2.end()) {
                double temp = offspring1[index].horizon_score;
                offspring1[index].horizon_score = it->horizon_score + dis_move(gen);
                it->horizon_score = temp + dis_move(gen);
            }
        }
    }

    return std::make_pair(offspring1, offspring2);
}

std::vector<Horizon> mutation(const std::vector<Horizon>& individual, int n_biostrat) {
    std::vector<Horizon> mutated_individual = individual;
    std::uniform_int_distribution<int> index_dist(0, int(mutated_individual.size() - 1));
    std::uniform_int_distribution<int> biostrat_dist(0, n_biostrat - 1);
    std::uniform_real_distribution<> dis_move(-1.0/individual.size(), 1.0/individual.size());

    // 扰动
    int num_swaps = 1 + index_dist(gen) % 1; // 在 1 到 3 之间随机选择扰动次数
    for (int i = 0; i < num_swaps; ++i) {
        int index1 = index_dist(gen);
        int index2 = index_dist(gen);
        while (index1 == index2) {
            index2 = index_dist(gen);
        }
        std::swap(mutated_individual[index1].horizon_score, mutated_individual[index2].horizon_score);
        mutated_individual[index1].horizon_score += dis_move(gen);
        mutated_individual[index2].horizon_score += dis_move(gen);
    }

    return mutated_individual;
}

//std::vector<Horizon> mutation(const std::vector<Horizon>& horizons, int n_biostrat) {
//    std::vector<Horizon> mutated_horizons = horizons;
//    int num_sections = 0;
//    int num_horizons = horizons.size();
//    for (const Horizon& horizon : horizons) {
//        if (horizon.section_number > num_sections) {
//            num_sections = horizon.section_number;
//        }
//    }
//    std::random_device rd;
//    std::mt19937 gen(rd());
//    std::uniform_real_distribution<> dis(0.0, 1.0);
//    std::uniform_int_distribution<> dis_int(1, num_sections);
//    std::uniform_real_distribution<> dis_move(-0.1, 0.1);
//    std::uniform_real_distribution<> dis_scale(-0.05, 0.05);
//    std::uniform_real_distribution<> dis_gap(-0.59, 5.0);
//
//    for (size_t i = 0; i < mutated_horizons.size(); ++i) {
//        double pmove = dis(gen);
//        int psec = dis_int(gen);
//        // 向上/向下移动
//        if (pmove < 0.2) {
//            double random_move = dis_move(gen);
//            for (Horizon& horizon : mutated_horizons) {
//                if (horizon.section_number == psec) {
//                    horizon.horizon_score += random_move;
//                }
//            }
//        }
//        // 扩张/收缩
//        else if (pmove < 0.4) {
//            double random_scale = dis_scale(gen) + 1.0;
//            double sum = 0.0;
//            int count = 0;
//            for (const Horizon& horizon : mutated_horizons) {
//                if (horizon.section_number == psec) {
//                    sum += horizon.horizon_score;
//                    count++;
//                }
//            }
//            double avg = sum / count;
//            for (Horizon& horizon : mutated_horizons) {
//                if (horizon.section_number == psec) {
//                    horizon.horizon_score = (horizon.horizon_score - avg) * random_scale + avg;
//                }
//            }
//        } else if (pmove < 0.6) {
//            // 插入/移除gap
//            int count = 0;
//            for (const Horizon& horizon : mutated_horizons) {
//                if (horizon.section_number == psec) {
//                    count++;
//                }
//            }
//            while (count < 3) {
//                psec = dis_int(gen);
//                count = 0;
//                for (const Horizon& horizon : mutated_horizons) {
//                    if (horizon.section_number == psec) {
//                        count++;
//                    }
//                }
//            }
//            std::uniform_int_distribution<> dis_breakpoint(2, count - 1);
//            int breakpoint = dis_breakpoint(gen);
//
//            double gap = 0.0;
//            int current_count = 0;
//            for (int k = 0; k < mutated_horizons.size(); ++k) {
//                if (mutated_horizons[k].section_number == psec) {
//                    current_count++;
//                    if (current_count == breakpoint + 1) {
//                        gap = (mutated_horizons[k].horizon_score - mutated_horizons[k - 1].horizon_score) * dis_gap(gen);
//                        break;
//                    }
//                }
//            }
//            current_count = 0;
//            for (Horizon& horizon : mutated_horizons) {
//                if (horizon.section_number == psec) {
//                    current_count++;
//                    if (current_count <= breakpoint + 1) {
//                        horizon.horizon_score += gap;
//                    }
//                }
//            }
//        } else if (pmove < 0.8) {
//            // dogleg
//            double shval = dis_move(gen) + 1.0;
//            std::vector<bool> ps(mutated_horizons.size());
//            for (size_t i = 0; i < mutated_horizons.size(); ++i) {
//                ps[i] = mutated_horizons[i].section_number == psec;
//            }
//            while (std::count(ps.begin(), ps.end(), true) < 3) {
//                psec = dis_int(gen);
//                for (size_t i = 0; i < mutated_horizons.size(); ++i) {
//                    ps[i] = mutated_horizons[i].section_number == psec;
//                }
//            }
//            std::vector<size_t> w;
//            for (size_t i = 0; i < ps.size(); ++i) {
//                if (ps[i]) {
//                    w.push_back(i);
//                }
//            }
//            std::uniform_int_distribution<> dis_breakpt(2, static_cast<double>(w.size()) - 1);
//            int breakpt = dis_breakpt(gen);
//            std::vector<double> gapval(w.size() - 1);
//            for (size_t i = 0; i < gapval.size(); ++i) {
//                gapval[i] = mutated_horizons[w[i + 1]].horizon_score - mutated_horizons[w[i]].horizon_score;
//            }
//            double upchoice = dis(gen);
//            if (upchoice > 0.5) {
//                for (size_t i = breakpt; i < w.size() - 1; ++i) {
//                    gapval[i] *= shval;
//                }
//            }
//            else {
//                for (size_t i = 0; i < breakpt; ++i) {
//                    gapval[i] *= shval;
//                }
//            }
//            std::vector<double> newval(w.size());
//            newval[0] = mutated_horizons[w[0]].horizon_score;
//            for (size_t i = 1; i < newval.size(); ++i) {
//                newval[i] = newval[i - 1] + gapval[i - 1];
//            }
//            for (size_t i = 0; i < w.size(); ++i) {
//                mutated_horizons[w[i]].horizon_score = newval[i];
//            }
//        } else {
//            // 插入乱序程序的策略
//            std::uniform_int_distribution<> dis_target(1, num_horizons);
//            int target = dis_target(gen);
//            std::uniform_int_distribution<> dis_dmove(0, 1);
//            int dmove = dis_dmove(gen);
//            int nmove = std::ceil(std::abs(std::normal_distribution<double>(0, 4)(gen)));
//            nmove = 1;
//
//            if (dmove == 0) {
//                int startsection = mutated_horizons[target - 1].section_number;
//                while (nmove > 0) {
//                    if (target == num_horizons) {
//                        nmove = 0;
//                    } else {
//                        if (mutated_horizons[target].section_number == startsection) {
//                            nmove = 0;
//                        } else {
//                            double temp = mutated_horizons[target].horizon_score;
//                            mutated_horizons[target].horizon_score = mutated_horizons[target - 1].horizon_score;
//                            mutated_horizons[target - 1].horizon_score = temp;
//                            target += 1;
//                            nmove -= 1;
//                        }
//                    }
//                }
//            } else {
//                while (nmove > 0) {
//                    if (target == 1) {
//                        nmove = 0;
//                    } else {
//                        if (mutated_horizons[target - 2].section_number == mutated_horizons[target - 1].section_number) {
//                            target -= 1;
//                        } else {
//                            double temp = mutated_horizons[target - 2].horizon_score;
//                            mutated_horizons[target - 2].horizon_score = mutated_horizons[target - 1].horizon_score;
//                            mutated_horizons[target - 1].horizon_score = temp;
//                            target -= 1;
//                            nmove -= 1;
//                        }
//                    }
//                }
//            }
//        }
//    }
//    std::sort(mutated_horizons.begin(), mutated_horizons.end(), [](const Horizon& a, const Horizon& b) { return a.horizon_score < b.horizon_score; });
//    if (calculate_fitness(mutated_horizons, n_biostrat) <= calculate_fitness(horizons, n_biostrat)) {
//        std::cout << "Current Best Fitness: " << calculate_fitness(mutated_horizons, n_biostrat) << std::endl;
//        return mutated_horizons;
//    } else {
//        return horizons;
//    }
//}

std::vector<Horizon> HorizonAnneal(std::vector<Horizon>& horizons, const PenaltyParameters& penalty_parameters, const int population_size, const int num_generations, const double crossover_rate, const double mutation_rate) {
    // Sort horizons by horizon_score
    std::sort(horizons.begin(), horizons.end(), [](const Horizon& a, const Horizon& b) { return a.horizon_score < b.horizon_score; });

    // Calculate min and max horizon_score
    double min_horizon_score = horizons.front().horizon_score;
    double max_horizon_score = horizons.back().horizon_score;

    // Normalize horizon_score for each horizon
    for (Horizon& horizon : horizons) {
        horizon.horizon_score = (horizon.horizon_score - min_horizon_score) / (max_horizon_score - min_horizon_score);
    }
    
    // 拷贝传入参数horizons至d3
    std::vector<Horizon> d3 = horizons;
    
    // 初始化种群
    std::vector<Chromosome> population;
    for (int i = 0; i < population_size; ++i) {
        population.push_back(init_chromosome(d3));
    }
    
    // 计算适应度
    double fitness_old = 0;
    for (Chromosome& chromosome : population) {
        chromosome.fitness = calculate_fitness(chromosome.horizons, penalty_parameters.n_biostrat);
        fitness_old += chromosome.fitness;
    }

    // 遗传算法主循环
    for (int i = 0; i < num_generations; ++i) {

        // 选择（锦标赛策略）：从population中随机挑选至mating_pool，size不变
        std::vector<Chromosome> mating_pool = selection(population, int(1 * population_size / 10));

        // 交叉与突变
        std::vector<Chromosome> new_population;
        for (int j = 0; j < population_size; j += 2) {
            Chromosome parent1 = mating_pool[j];
            Chromosome parent2 = mating_pool[j + 1];

            // 交叉：任选
            if (dis(gen) < crossover_rate) {
                std::tie(parent1.horizons, parent2.horizons) = crossover(parent1.horizons, parent2.horizons);
            }

            // 突变
            if (dis(gen) < mutation_rate) {
                parent1.horizons = mutation(parent1.horizons, penalty_parameters.n_biostrat);
            }
            if (dis(gen) < mutation_rate) {
                parent2.horizons = mutation(parent2.horizons, penalty_parameters.n_biostrat);
            }

            new_population.push_back(parent1);
            new_population.push_back(parent2);

        }
        for (Chromosome& chromosome : new_population) {
            std::sort(chromosome.horizons.begin(), chromosome.horizons.end(), [](const Horizon& a, const Horizon& b) { return a.horizon_score < b.horizon_score; });
        }
        // 计算总适应度
        int fitness_sum = 0;
        for (Chromosome& chromosome : new_population) {
            chromosome.fitness = calculate_fitness(chromosome.horizons, penalty_parameters.n_biostrat);
            fitness_sum += chromosome.fitness;
        }
        // 计算当前代的最佳适应度
        auto best_chromosome = std::min_element(new_population.begin(), new_population.end(), [](const Chromosome& a, const Chromosome& b) {
            return a.fitness < b.fitness;
        });
        // 计算前一代的最佳适应度
        auto best_chromosome_old = std::min_element(population.begin(), population.end(), [](const Chromosome& a, const Chromosome& b) {
            return a.fitness < b.fitness;
        });
        if (best_chromosome->fitness < best_chromosome_old->fitness) {
            population = new_population;
            fitness_old = fitness_sum;
            std::cout << "Generation: " << i + 1 << " | Best fitness: " << best_chromosome->fitness << std::endl;
        } else {
            std::cout << "Generation: " << i + 1 << " | Best fitness: " << best_chromosome_old->fitness << std::endl;
        }
        for (Chromosome& chromosome : population) {
            for (int i = 0; i < chromosome.horizons.size(); i++) {
                chromosome.horizons[i].horizon_score = i / double(chromosome.horizons.size());
            }
        }
    }

    // 从最终种群中获取最佳染色体
    auto best_chromosome = std::min_element(population.begin(), population.end(), [](const Chromosome& a, const Chromosome& b) {
        return a.fitness < b.fitness;
    });

    // 更新解决方案
    d3 = best_chromosome->horizons;
    return d3;
}

std::string getCurrentTimeString() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);

    std::tm local_tm;
    localtime_r(&now_time, &local_tm);

    std::ostringstream oss;
    oss << std::put_time(&local_tm, "%Y%m%d_%H%M%S");
    return oss.str();
}

std::string generateFilename(const std::string& prefix, const std::string& extension) {
    std::ostringstream oss;
    oss << prefix << "_" << getCurrentTimeString() << extension;
    return oss.str();
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    
    const std::string file_path = "/Users/qianxian/PycharmProjects/HA/dataset/riley_62_for_R.csv";
    std::vector<Horizon> horizons = read_csv_data(file_path, header_line);
    PenaltyParameters penalty_parameters = initialize_penalty_parameters();
    
    int num_generations = 1000; // outer_num
    int population_size = 1000; // inner_num

    double crossover_rate = 0.99;
    double mutation_rate = 0.99;
    std::vector<Horizon> result = HorizonAnneal(horizons, penalty_parameters, population_size, num_generations, crossover_rate, mutation_rate);
    std::string prefix = "/Users/qianxian/PycharmProjects/HA/output/GA_output";
    std::string extension = ".csv";
    std::string output_file_path = generateFilename(prefix, extension);
    save_to_csv(result, output_file_path);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "\nElapsed time: " << elapsed.count() << " seconds." << std::endl;
    return 0;
}
