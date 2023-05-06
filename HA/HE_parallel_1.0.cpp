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
#include <mpi.h>
#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>

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

namespace boost {
    namespace serialization {

        template<class Archive>
        void serialize(Archive& ar, Horizon& h, const unsigned int version) {
            ar& h.section_name;
            ar& h.horizon_score;
            ar& h.section_number;
            ar& h.horizon_number;
            ar& h.horizon_height;
            ar& h.presence_absence_data;
        }
        template<class Archive>
        void serialize(Archive& ar, Chromosome& c, const unsigned int version) {
            ar& c.horizons;
            ar& c.fitness;
        }
    } // namespace serialization
} // namespace boost

std::vector<Horizon> read_csv_data(const std::string& file_path, std::string& header_line);
void save_to_csv(const std::vector<Horizon>& horizons, const std::string& file_path, const std::string& header_line);
PenaltyParameters initialize_penalty_parameters();
std::vector<Horizon> init_horizons(std::vector<Horizon>& horizons);
Chromosome init_chromosome(const std::vector<Horizon>& d3);
int calculate_penalty(const std::vector<Horizon>& horizons, int n_biostrat);
double calculate_fitness(const std::vector<Horizon>& horizons, int n_biostrat);
std::vector<Chromosome> selection(const std::vector<Chromosome>& population, int tournament_size);
std::tuple<std::vector<Horizon>, std::vector<Horizon>> crossover(const std::vector<Horizon>& parent1, const std::vector<Horizon>& parent2);
std::vector<Horizon> mutation(const std::vector<Horizon>& individual, int n_biostrat);
void repair_chromosome(std::vector<Horizon>& chromosome);
std::string getCurrentTimeString();
std::string generateFilename(const std::string& prefix, const std::string& extension);
void update_population_with_best_solution(std::vector<Chromosome>& population, const Chromosome& received_best_solution);
void update_best_solution_library(std::vector<Chromosome>& best_solution_library, const Chromosome& received_solution, size_t LIBRARY_SIZE);

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);

int main(int argc, char* argv[]) {
    // MPI 初始化
    MPI_Init(&argc, &argv);
    auto start_time = std::chrono::high_resolution_clock::now();
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int max_rank = size - 1;

    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;

    // 从csv文件读取数据
    const std::string file_path = "HA_Fan_et_al_2013.csv";
    std::string header_line;
    std::vector<Horizon> horizons = read_csv_data(file_path, header_line);

    srand(time(NULL) * rank); // 设置随机数种子
    const int N_HORZIONS = horizons.size(); // 设置水平层数量
    const int N_BIOSTRAT = horizons[0].presence_absence_data.size(); // 设置生物数量
    const int num_generations = 1000; // outer_num(最大迭代次数)
    const int population_size = 100; // inner_num
    const int exchange_interval = 5; // 设置交换间隔
    const int library_size = 100; // 设置最优解库大小
    double crossover_rate = 0.99;
    double mutation_rate = 0.99;

    //// 初始化从文件中读取的序列
    //// 根据horizon_score进行排序
    //std::sort(horizons.begin(), horizons.end(), [](const Horizon& a, const Horizon& b) { return a.horizon_score < b.horizon_score; });
    //// 计算最大和最小horizon_score
    //double min_horizon_score = horizons.front().horizon_score;
    //double max_horizon_score = horizons.back().horizon_score;
    //// 归一化horizon_score
    //for (Horizon& horizon : horizons) {
    //    horizon.horizon_score = (horizon.horizon_score - min_horizon_score) / (max_horizon_score - min_horizon_score);
    //}

    // 设置消息标识
    const int MSG_REQUEST_BEST_SOLUTION = 13;
    const int MSG_SEND_BEST_SOLUTION = 14;

    std::vector<Chromosome> best_solutions(library_size); // 创建最优解库
    if (rank == 0) { // 主进程（负责维护最优解库）
        MPI_Status status;

        // 初始化解库
        for (int i = 0; i < library_size; ++i) {
            best_solutions[i] = init_chromosome(horizons);
        }

        int count = 0;
        while (count < 2 * num_generations) {
            count++;
            int msg_type;
            MPI_Recv(&msg_type, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            int sender_rank = status.MPI_SOURCE; // 获取对应子进程序号

            if (msg_type == MSG_REQUEST_BEST_SOLUTION) { // 发送解
                // 向子进程发送确认接收标识
                MPI_Send(&msg_type, 1, MPI_INT, sender_rank, 0, MPI_COMM_WORLD);

                // 从解库中随机挑选一个较优解发送给子进程
                int target_rank = rand() % library_size;
                Chromosome best_solution = best_solutions[target_rank];
                //std::vector<int> serialized_data(serialize_chromosome_size(best_solution));
                //serialized_data = serialize_chromosome(best_solution);
                //MPI_Send(serialized_data.data(), serialized_data.size(), MPI_INT, sender_rank, 0, MPI_COMM_WORLD);
                world.send(sender_rank, 0, best_solution);
            }
            else if (msg_type == MSG_SEND_BEST_SOLUTION) { // 接收解
                // 向子进程发送确认接收标识
                MPI_Send(&msg_type, 1, MPI_INT, sender_rank, 0, MPI_COMM_WORLD);

                // 接收子进程发送的较优解
                //std::vector<int> serialized_data(serialize_chromosome_size(best_solutions[0]));
                //MPI_Recv(serialized_data.data(), serialized_data.size(), MPI_INT, sender_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                //// 反序列化接收到的染色体
                //Chromosome received_chromosome;
                //deserialize_chromosome(serialized_data, received_chromosome, N_HORZIONS, N_BIOSTRAT);

                Chromosome received_chromosome;
                world.recv(sender_rank, 0, received_chromosome);

                // 使用接收到的当前最优解更新全局最优解库
                update_best_solution_library(best_solutions, received_chromosome, library_size);
                // 计算解库中最优解
                auto best_solution = std::min_element(best_solutions.begin(), best_solutions.end(), [](const Chromosome& a, const Chromosome& b) { return a.fitness < b.fitness; });
                std::cout << "rank:" << sender_rank << " | Current fitness:" << received_chromosome.fitness << " | Best fitness:" << best_solution->fitness << std::endl;
            }
        }
    }
    else { // 子进程（负责解的搜索）
        // 初始化种群
        std::vector<Horizon> d3 = horizons;
        std::vector<Chromosome> population;
        for (int i = 0; i < population_size; ++i) {
            population.push_back(init_chromosome(d3));
        }
        // 更新所有个体适应度并计算最佳适应度
        double best_fitness = std::numeric_limits<double>::infinity();
        for (Chromosome& chromosome : population) {
            chromosome.fitness = calculate_fitness(chromosome.horizons, N_BIOSTRAT);
            if (chromosome.fitness < best_fitness) {
                best_fitness = chromosome.fitness;
            }
        }
        std::cout << "rank:" << rank << " | Starting penalty: " << best_fitness << "" << std::endl;

        for (int generation = 0; generation < num_generations; ++generation) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(0.0, 1.0);
            // 执行遗传算法的一次迭代
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
                    parent1.horizons = mutation(parent1.horizons, N_BIOSTRAT);
                }
                if (dis(gen) < mutation_rate) {
                    parent2.horizons = mutation(parent2.horizons, N_BIOSTRAT);
                }
                new_population.push_back(parent1);
                new_population.push_back(parent2);
            }
            for (Chromosome& chromosome : new_population) {
                // 修复个体以满足约束条件
                repair_chromosome(chromosome.horizons);
                std::sort(chromosome.horizons.begin(), chromosome.horizons.end(), [](const Horizon& a, const Horizon& b) { return a.horizon_score < b.horizon_score; });
            }
            // 计算并更新适应度
            for (Chromosome& chromosome : new_population) {
                chromosome.fitness = calculate_fitness(chromosome.horizons, N_BIOSTRAT);
            }
            // 计算当前代的最佳适应度
            auto current_chromosome = std::min_element(new_population.begin(), new_population.end(), [](const Chromosome& a, const Chromosome& b) { return a.fitness < b.fitness; });
            // 更新子种群
            population = new_population;
            best_fitness = std::min(best_fitness, current_chromosome->fitness);
            // 归一化horizon_score
            for (Chromosome& chromosome : population) {
                for (int i = 0; i < chromosome.horizons.size(); i++) {
                    chromosome.horizons[i].horizon_score = i / double(chromosome.horizons.size());
                }
            }
            // 各子进程轮流向主进程发送/请求解
            if ((generation % max_rank + 1) == rank) {
                // 向主进程发送当前子进程的最优解
                int msg_type = MSG_SEND_BEST_SOLUTION;
                MPI_Send(&msg_type, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                
                // 接收主进程的确认接收标识
                MPI_Recv(&msg_type, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // 发送当前子进程的最优解
                Chromosome best_solution;
                best_solution.horizons = current_chromosome->horizons;
                best_solution.fitness = current_chromosome->fitness;
                world.send(0, 0, best_solution);

                // 请求主进程发送一个较优解
                msg_type = MSG_REQUEST_BEST_SOLUTION;
                MPI_Send(&msg_type, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

                // 接收主进程的确认接收标识 
                MPI_Recv(&msg_type, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                Chromosome received_chromosome;
                world.recv(0, 0, received_chromosome);

                // 使用接收到的较优解更新子进程的种群
                update_population_with_best_solution(population, received_chromosome);
            }
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    long long max_duration;
    MPI_Reduce(&duration, &max_duration, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        auto best_solution = std::min_element(best_solutions.begin(), best_solutions.end(), [](const Chromosome& a, const Chromosome& b) { return a.fitness < b.fitness; });
        std::string prefix = "GA_output";
        std::string extension = ".csv";
        std::string output_file_path = generateFilename(prefix, extension);
        save_to_csv(best_solution->horizons, output_file_path, header_line);
        std::cout << "Total time elapsed: " << max_duration << " seconds" << std::endl;
    }
    MPI_Finalize();
    return 0;
}

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

void save_to_csv(const std::vector<Horizon>& horizons, const std::string& file_path, const std::string& header_line) {
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

    //params.n_biostrat = 62;
    params.n_biostrat = 146;
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

std::vector<Horizon> init_horizons(std::vector<Horizon>& horizons) {
    return horizons;
}

Chromosome init_chromosome(const std::vector<Horizon>& d3) {
    Chromosome chromosome;
    chromosome.horizons = d3;
    // 为每个section内的horizon分配新的score
    for (size_t i = 0; i < chromosome.horizons.size();) {
        size_t section_start = i;
        int current_section = chromosome.horizons[i].section_number;

        // 找到当前section的结束位置
        while (i < chromosome.horizons.size() && chromosome.horizons[i].section_number == current_section) {
            ++i;
        }

        // 分配新的score
        double last_score = 0.0;
        for (size_t j = section_start; j < i; ++j) {
            double delta_score = dis(gen);
            chromosome.horizons[j].horizon_score = last_score + delta_score;
            last_score = chromosome.horizons[j].horizon_score;
        }
    }
    // 根据horizon_score进行排序
    std::sort(chromosome.horizons.begin(), chromosome.horizons.end(), [](const Horizon& a, const Horizon& b) { return a.horizon_score < b.horizon_score; });
    // 计算最大和最小horizon_score
    double min_horizon_score = chromosome.horizons.front().horizon_score;
    double max_horizon_score = chromosome.horizons.back().horizon_score;
    // 归一化horizon_score
    for (Horizon& horizon : chromosome.horizons) {
        horizon.horizon_score = (horizon.horizon_score - min_horizon_score) / (max_horizon_score - min_horizon_score);
    }
    // 计算更新后解的penalty
    chromosome.fitness = calculate_fitness(chromosome.horizons, chromosome.horizons[0].presence_absence_data.size());
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

    std::uniform_int_distribution<> dis_cross(0, int(parent1.size() - 1));
    std::uniform_real_distribution<> dis_move(-1.0 / parent1.size(), 1.0 / parent1.size());

    std::unordered_set<int> selected_indices;
    int crossover_size = int(1 * parent1.size() / 2);

    while (selected_indices.size() < crossover_size) {
        int index = dis_cross(gen);
        if (selected_indices.find(index) == selected_indices.end()) {
            selected_indices.insert(index);

            int section_number = offspring1[index].section_number;
            int horizon_number = offspring1[index].horizon_number;

            // Find the corresponding horizon in offspring2
            auto it = std::find_if(offspring2.begin(), offspring2.end(), [section_number, horizon_number](const Horizon& horizon) { return (horizon.section_number == section_number) & (horizon.horizon_number == horizon_number); });

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

void update_population_with_best_solution(std::vector<Chromosome>& population, const Chromosome& received_best_solution) {
    // 找到种群中最差个体的索引
    int worst_solution_index = 0;
    double worst_fitness = population[0].fitness;

    for (size_t i = 1; i < population.size(); ++i) {
        if (population[i].fitness > worst_fitness) {
            worst_fitness = population[i].fitness;
            worst_solution_index = i;
        }
    }

    // 用接收到的较优解替换最差个体
    population[worst_solution_index] = received_best_solution;
}

void update_best_solution_library(std::vector<Chromosome>& best_solution_library, const Chromosome& received_solution, size_t LIBRARY_SIZE) {
    // 计算接收到的解的适应度
    double received_fitness = received_solution.fitness;

    // 寻找库中适应度较差的解的索引
    int worst_solution_index = -1;
    double worst_fitness = -1;

    // 遍历库中的解，寻找适应度较差的解
    for (size_t i = 0; i < best_solution_library.size(); ++i) {
        if (worst_solution_index == -1 || best_solution_library[i].fitness > worst_fitness) {
            worst_solution_index = i;
            worst_fitness = best_solution_library[i].fitness;
        }
    }

    // 如果接收到的解的适应度好于库中最差解的适应度，用接收到的解替换库中最差解
    if (received_fitness < worst_fitness) {
        best_solution_library[worst_solution_index] = received_solution;
    }
    // 如果库未满，则直接将接收到的解添加到库中
    else if (best_solution_library.size() < LIBRARY_SIZE) {
        best_solution_library.push_back(received_solution);
    }
}
