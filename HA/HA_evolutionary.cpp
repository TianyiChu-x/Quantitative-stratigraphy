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
#include <iomanip>
#include <map>

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
std::vector<Horizon> read_csv_data(const std::string& file_path, std::string& header_line) {
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
    // params.n_biostrat = 4522;
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

    return selected_population;
}

std::tuple<std::vector<Horizon>, std::vector<Horizon>> crossover(const std::vector<Horizon>& parent1, const std::vector<Horizon>& parent2) {
    std::vector<Horizon> offspring1 = parent1;
    std::vector<Horizon> offspring2 = parent2;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, int(parent1.size() - 1));
    std::uniform_real_distribution<> dis_move(-1.0 / parent1.size(), 1.0 / parent1.size());

    std::unordered_set<int> selected_indices;
    int crossover_size = int(1 * parent1.size() / 2);

    while (selected_indices.size() < crossover_size) {
        int index = dis(gen);
        if (selected_indices.find(index) == selected_indices.end()) {
            selected_indices.insert(index);

            int section_number = offspring1[index].section_number;
            int horizon_number = offspring1[index].horizon_number;

            // Find the corresponding horizon in offspring2
            auto it = std::find_if(offspring2.begin(), offspring2.end(), [section_number, horizon_number](const Horizon& horizon) { return horizon.section_number == section_number & horizon.horizon_number == horizon_number; });

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
    std::uniform_real_distribution<> dis_move(-1.0 / individual.size(), 1.0 / individual.size());

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

void repair_chromosome(std::vector<Horizon>& chromosome) {
    // 按 section_number 对 chromosome 进行分组
    std::map<int, std::vector<Horizon*>> sections;
    for (auto& horizon : chromosome) {
        sections[horizon.section_number].push_back(&horizon);
    }

    // 对于每个 section，按照 horizon_number 对其进行排序
    for (auto& section : sections) {
        std::sort(section.second.begin(), section.second.end(),
            [](const Horizon* a, const Horizon* b) {
                return a->horizon_number < b->horizon_number;
            });

        // 确保 horizon_score 按照 horizon_number 递增
        for (size_t i = 1; i < section.second.size(); ++i) {
            if (section.second[i]->horizon_score < section.second[i - 1]->horizon_score) {
                section.second[i]->horizon_score = section.second[i - 1]->horizon_score;
            }
        }
    }
}

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

    // 计算最佳适应度
    double best_fitness = std::numeric_limits<double>::infinity();
    for (Chromosome& chromosome : population) {
        chromosome.fitness = calculate_fitness(chromosome.horizons, penalty_parameters.n_biostrat);
        if (chromosome.fitness < best_fitness) {
            best_fitness = chromosome.fitness;
        }
    }
    std::cout << "Starting penalty: " << best_fitness << "" << std::endl;

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
            // 修复个体以满足约束条件
            repair_chromosome(chromosome.horizons);
            std::sort(chromosome.horizons.begin(), chromosome.horizons.end(), [](const Horizon& a, const Horizon& b) { return a.horizon_score < b.horizon_score; });
        }

        // 计算适应度
        for (Chromosome& chromosome : new_population) {
            chromosome.fitness = calculate_fitness(chromosome.horizons, penalty_parameters.n_biostrat);
        }

        // 计算当前代的最佳适应度
        auto current_chromosome = std::min_element(new_population.begin(), new_population.end(), [](const Chromosome& a, const Chromosome& b) { return a.fitness < b.fitness; });

        // 更新种群
        population = new_population;
        best_fitness = std::min(best_fitness, current_chromosome->fitness);
        std::cout << "Generation: " << i + 1 << " | Best fitness: " << best_fitness << " | Current fitness:" << current_chromosome->fitness << std::endl;
        for (Chromosome& chromosome : population) {
            for (int i = 0; i < chromosome.horizons.size(); i++) {
                chromosome.horizons[i].horizon_score = i / double(chromosome.horizons.size());
            }
        }
    }

    // 从最终种群中获取最佳染色体
    auto best_chromosome = std::min_element(population.begin(), population.end(), [](const Chromosome& a, const Chromosome& b) { return a.fitness < b.fitness; });

    // 更新解决方案
    d3 = best_chromosome->horizons;
    return d3;
}

std::string getCurrentTimeString() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);

    std::tm local_tm;
    #ifdef _WIN32
    localtime_s(&local_tm, &now_time);
    #else
    localtime_r(&now_time, &local_tm);
    #endif

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

    const std::string file_path = "riley_62_for_R.csv";
    std::vector<Horizon> horizons = read_csv_data(file_path, header_line);
    PenaltyParameters penalty_parameters = initialize_penalty_parameters();

    int num_generations = 1000; // outer_num
    int population_size = 1000; // inner_num

    double crossover_rate = 0.99;
    double mutation_rate = 0.99;
    std::vector<Horizon> result = HorizonAnneal(horizons, penalty_parameters, population_size, num_generations, crossover_rate, mutation_rate);
    std::string prefix = "GA_output";
    std::string extension = ".csv";
    std::string output_file_path = generateFilename(prefix, extension);
    save_to_csv(result, output_file_path);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "\nElapsed time: " << elapsed.count() << " seconds." << std::endl;
    return 0;
}
