#include <iostream>
#include <vector>
#include <unordered_set>
#include <set>
#include <chrono>
#include <queue>
#include <utility>
#include <new>
#include <algorithm>
#include <stack>
#include <random>
#include <cmath>
#include <fstream>
/*
used for plotting routes
#include <boost/tuple/tuple.hpp>
#include "gnuplot-iostream.h"
*/

// writes a vector of pairs to a given file name
void WriteToFile(std::string file_name, std::vector< std::pair<double, double> > array) {
  std::ofstream fout(file_name);
  for(int i = 0;i < array.size(); i++) {
    fout<<array[i].first<<','<<array[i].second<<'\n';
  }
}

// represents each solution/chromosome
struct State{
  std::vector< std::pair <double,double> > permutation;
  double cost;
  int operator ==(State b) const {
    return (permutation == b.permutation) && (cost == b.cost);
  }
};

// a hash function for random_states
struct StateHash {
    size_t operator()(const State &v) const {
        size_t seed = 0;
      	for(auto t = v.permutation.begin();t != v.permutation.end();t++) {
            size_t h1 = std::hash<double>()((*t).first);
            size_t h2 = std::hash<double>()((*t).second);
            seed ^= (h1 ^ (h2 << 1));
        }
        return seed;
    }
};

// a hash function for random_states used for unordered_set
struct PairHash {
    size_t operator()(const std::pair<double,double> &v) const {
        size_t seed = 0;
        size_t h1 = std::hash<double>()(v.first);
        size_t h2 = std::hash<double>()(v.second);
        seed = (h1 ^ (h2 << 1));
        return seed;
    }
};

// two states are compared by their path cost
bool CompareState(State first, State second) {
    return (first.cost < second.cost);
}
// forward declaring functions
double StateCost(State);

State BeamSearch(State);

State SimulatedAnnealing(State);

State GeneticAlgorithm(State);

//  lists to store best, worst and best of each generation
std::vector<std::pair<double,double>> best_of_generation;

std::vector<std::pair<double,double>> worst_of_generation;

std::vector<std::pair<double,double>> average_of_generation;

// probability bins for rank selection
std::vector< int > pool_probability_bins;

int main() {
  // used for plotting route
  //Gnuplot gp;
  // recieve city Coordinates
  std::vector<std::pair<double, double> > points;
	int n;
	double x,y;
	std::string method;
	State nodes;
	State result;
	std::cout<<"Enter X and Y Coordinates of Initial Node:\n";
  std::cin>>x>>y;
  nodes.permutation.push_back(std::make_pair(x,y));
	std::cout<<"Enter Number of Nodes other than the Initial Node:\n";
	std::cin>>n;
    std::cout<<"Enter X and Y Coordinates of the Nodes:\n";
    for(int i = 0; i < n;i++) {
		std::cin>>x>>y;
		nodes.permutation.push_back(std::make_pair(x,y));
	}
  // create initial state / permutation
	nodes.cost = StateCost(nodes);
  // run the selected method
	std::cout<<"Enter Beam, SA or GA\n";
	std::cin>>method;
  // start timer
  auto start = std::chrono::high_resolution_clock::now();
	if(method == "Beam") {
		result = BeamSearch(nodes);
	}
	if(method == "SA") {
		result = SimulatedAnnealing(nodes);
	}
	if(method == "GA") {
		result = GeneticAlgorithm(nodes);
	}
  // stop timer and print the time it took and the best route cost
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double>elapsed = finish - start;
	std::cout<<"Duration: "<<elapsed.count()<<'\n'<<result.cost;

  // plot best route
  /*
  for(int i = 0;i<result.permutation.size();i++) {
        points.push_back(result.permutation[i]);
	}
	points.push_back(result.permutation[0]);
*/
  //gp << "set xrange [0:1000]\nset yrange [0:100]\n";
	//gp << "plot" << gp.file1d(average_of_generation) << "with lines title 'average',"<<std::endl;
	//gp << "plot" << gp.file1d(best_of_generation) << "with lines title 'best',"<<std::endl;
	//gp << "plot" << gp.file1d(worst_of_generation) << "with lines title 'worst',"<<std::endl;

  // write generation results to files
  //WriteToFile("average_of_generation1000k1000g.txt",average_of_generation);
  //WriteToFile("best_of_generation1000k1000g.txt",best_of_generation);
  //WriteToFile("worst_of_generation1000k1000g.txt",worst_of_generation);
  return 0;
}

int Factorial(int num) {
    int res = 1;
    for(int i = num;i>1;i--) {
        res*=i;
    }
    return res;
}

// computes cost of permutation
double StateCost (State nodes) {
    double cost = 0;
    for(int i = 0;i<nodes.permutation.size()-1;i++) {
        cost += sqrt(pow((nodes.permutation[i].first - nodes.permutation[i+1].first),2) + pow((nodes.permutation[i].second - nodes.permutation[i+1].second),2));
    }
    cost += sqrt(pow((nodes.permutation[0].first - nodes.permutation[nodes.permutation.size()-1].first),2) +
                 pow((nodes.permutation[0].second - nodes.permutation[nodes.permutation.size()-1].second),2));
    return cost;
}

// a binary search to find index/bin which the input belongs to
int FindIndex(int input) {
  int l = 0;
  int r = pool_probability_bins.size()-1;
  int m;
  while(l<r) {
    m = (l+r)/2;
    if(pool_probability_bins[m] == input) {
      break;
    }
    else if(input > pool_probability_bins[m]) {
      if(input < pool_probability_bins[m+1]) {
        break;
      }
      else {
        l = m+1;
      }
    }
    else {
      if(input > pool_probability_bins[m-1]) {
        m--;
        break;
      }
      else {
        r = m-1;
      }
    }
  }
  return m;
}

// creates random permutations and adds greedy state to poputlation
std::vector < State > RandomPopulation(State nodes , int size_of_population = -1) {
	std::unordered_set < State , StateHash > random_states;
	State new_state;
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator (seed);
	if(size_of_population == -1) {
        size_of_population = std::min(std::max(1,int(1e5/pow(nodes.permutation.size(),2))),int((nodes.permutation.size()-1 > 9) ? 1e6 : Factorial(nodes.permutation.size()-1)));
	}
	else {
	    size_of_population = std::min(size_of_population,int((nodes.permutation.size()-1 > 9) ? 1e6 : Factorial(nodes.permutation.size()-1)));
	}
	for(int k = 0;k < size_of_population-1;k++) {
	    new_state = nodes;
	    for(int i = 1;i<nodes.permutation.size()-1;i++) {
	        std::uniform_int_distribution<int> distribution(i,nodes.permutation.size()-1);
	        int j = distribution(generator);
	        std::swap(new_state.permutation[i],new_state.permutation[j]);
	    }
	    new_state.cost = StateCost(new_state);
	    random_states.insert(new_state);
  }
  // computing greedy route/permutaion by finding closest city at each step
  std::set < int > visited;
  State greedy;
  visited.insert(0);
  int current = 0;
  greedy.permutation.push_back(nodes.permutation[0]);
  for(int i = 0; i < nodes.permutation.size()-1; i++){
    int closest = -1;
    double min_distnace = 99999;
    for(int j = 0; j < nodes.permutation.size()-1; j++){
      if(!visited.count(j)){
        if(pow((nodes.permutation[current].first - nodes.permutation[j].first),2) +
           pow((nodes.permutation[current].second - nodes.permutation[j].second),2) < min_distnace){
          min_distnace = pow((nodes.permutation[current].first - nodes.permutation[j].first),2) +
                        pow((nodes.permutation[current].second - nodes.permutation[j].second),2);
          closest = j;
         }
      }
    }
    visited.insert(closest);
    greedy.permutation.push_back(nodes.permutation[closest]);
    current = closest;
  }
  greedy.cost = StateCost(greedy);
  std::cerr<<greedy.cost<<'\n';
  std::vector < State > return_states;
  return_states.push_back(greedy);
  for(auto state = random_states.begin();state != random_states.end();state++) {
    return_states.push_back(*state);
  }
	return return_states;
}


// applies HillClimb on given state/permutation
// neighbours are produced by swapping two cities
State HillClimb (State nodes) {
    State min_neighbour;
    bool min_defined = false;
    do{
        if(min_defined == true) {
            nodes = min_neighbour;
            min_defined = false;
        }
        for(int i = 1;i<nodes.permutation.size();i++) {
            for(int j = i+1;j<nodes.permutation.size();j++) {
                State neighbour_state = nodes;
                swap(neighbour_state.permutation[i],neighbour_state.permutation[j]);
                neighbour_state.cost = StateCost(neighbour_state);
                if(min_defined == false) {
                    min_neighbour = neighbour_state;
                    min_defined = true;
                }
                else if(neighbour_state.cost < min_neighbour.cost) {
                    min_neighbour = neighbour_state;
                }
            }
        }
    }while(min_neighbour.cost < nodes.cost);
    return nodes;
}

// missleading name. it basically does HillClimb on each state in RandomPopulation
State BeamSearch(State nodes) {
    std::vector < State > initial_states = RandomPopulation(nodes,1);
    State final_state;
    bool final_defined = false;
    double final_state_cost = 0;
    for(auto state = initial_states.begin();state != initial_states.end();state++) {
        State local_min = HillClimb(*state);
        if(final_defined == false) {
            final_state = local_min;
            final_defined = true;
        }
        else if(local_min.cost < final_state.cost) {
            final_state = local_min;
        }
    }
    return final_state;
}

// a simple mutation function swapping two indexes
State RandomNeighbour(State nodes) {
  // choose two random numbers to swap
	State new_state;
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator (seed);
  new_state = nodes;
  std::uniform_int_distribution<int> distribution(1,nodes.permutation.size()-1);
  int j = distribution(generator);
  int i = distribution(generator);
  // swap chosen indexes
  std::swap(new_state.permutation[i],new_state.permutation[j]);
  new_state.cost = StateCost(new_state);
	return new_state;
}

// SimulatedAnnealing
State SimulatedAnnealing(State nodes) {
    double temperature = 400;
    double cooling_factor = 0.99999;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    while(temperature > 1e-9) {
        State neighbour = RandomNeighbour(nodes);
        if(neighbour.cost < nodes.cost) {
            nodes = neighbour;
        }
        else if(exp((neighbour.cost-nodes.cost)/temperature) < dis(gen)) {
             nodes = neighbour;
        }
        temperature *= cooling_factor;
    }
    return nodes;
}

// uses rank selection to generate breeding_pool
std::vector< State > RankSelection(std::vector< State > population,int kPopulationSize) {
  std::vector<State> breeding_pool;
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator (seed);
  std::uniform_int_distribution<int> distribution(0,(kPopulationSize*(kPopulationSize+1))/2);
  for(int i = 0; i<kPopulationSize;i++) {
    int j = distribution(generator);
    // finds index of chosen number in pool_probability_bins
    int chosen = FindIndex(j);
    breeding_pool.push_back(population[chosen]);
  }
  return breeding_pool;
}

// an order recombination based Cross Over function
State CrossOver(State parent1 ,State parent2) {
  // choose two random points for recombination
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator (seed);
  std::uniform_int_distribution<int> distribution(1,parent1.permutation.size()-1);
  std::unordered_set < std::pair<double,double> , PairHash > seen;
  int a = distribution(generator);
  int b = distribution(generator);
  // fill offspring with the a-b cut of first parent
  State new_state;
  for(int i = std::min(a,b);i<=std::max(a,b);i++) {
    new_state.permutation.push_back(parent1.permutation[i]);
    seen.insert(parent1.permutation[i]);
  }
  // fill the rest of offspring with rest of second parent
  int k=-1;
  for(int i = 0;i<std::min(a,b);i++) {
    while(seen.count(parent2.permutation[++k])) {}
    new_state.permutation.insert(new_state.permutation.begin(),parent2.permutation[k]);
  }
  for(int i = new_state.permutation.size();i<parent1.permutation.size();i++) {
    while(seen.count(parent2.permutation[++k])) {}
    new_state.permutation.push_back(parent2.permutation[k]);
  }
  new_state.cost = StateCost(new_state);
  return new_state;
}

State GeneticAlgorithm(State nodes) {
  const double kElitePercentage = 0.1;
  const double kMutationRate = 0.01;
  const int kGenerations = 1000;
  const int kPopulationSize = 500;
  std::vector < State > population = RandomPopulation(nodes, kPopulationSize);
  const int kEliteSize = int(kElitePercentage*population.size());
  State min_state;
  min_state.cost = 1000000;
  // rank based selection bins
  pool_probability_bins.push_back(0);
  for(int k = 0;k<kPopulationSize;k++) {
    int temp = pool_probability_bins[k];
    pool_probability_bins.push_back(temp + kPopulationSize - k);
  }

  // execute GeneticAlgorithm with specified number of generations
  for(int k = 0;k<kGenerations;k++) {
    std::vector < State > new_population;
    std::vector < State >  breeding_pool;
    std::vector < State >  offsprings;
    // sort current population
    std::sort(population.begin(),population.end(),CompareState);
    // save best and worst of population
    best_of_generation.push_back(std::make_pair(k, (*(population.begin())).cost));
    worst_of_generation.push_back(std::make_pair(k, (population[population.size()-1]).cost));

    double temp_avg = 0;
    int temp_counter = 1;
    // calculate average fitness of population
    for(auto it = population.begin();it != population.end();it++) {
      temp_avg = temp_avg / temp_counter * (temp_counter-1) + (*it).cost / temp_counter;
      temp_counter++;
    }
    // save average fitness of population
    average_of_generation.push_back(std::make_pair(k, temp_avg));
    // add elites to new population
    for(int i = 0;i<kEliteSize;i++) {
      new_population.push_back(population[i]);
    }
    // get breeding pool with RankSelection
    breeding_pool = RankSelection(population,kPopulationSize);
    for(int t = 0;t<population.size()-kEliteSize;t++) {
      // choose two random states from breeding_pool
      unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
      std::default_random_engine generator (seed);
      std::uniform_int_distribution<int> distribution(0,breeding_pool.size()-1);
      int j = distribution(generator);
      int i = distribution(generator);
      // create offspring of two parents using CrossOver
      State offspring = CrossOver(breeding_pool[i],breeding_pool[j]);
      offsprings.push_back(offspring);
    }
    // mutates an offspring with given probability
    for(auto it = offsprings.begin();it != offsprings.end();it++) {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<> dis(0.0, 1.0);
      if(dis(gen) < kMutationRate) {
        State mutant = RandomNeighbour(*it);
        new_population.push_back(mutant);
      }
      else {
        new_population.push_back(*it);
      }
    }
    breeding_pool.erase(breeding_pool.begin(),breeding_pool.end());
    for(int it = 0;it < population.size();it++) {
      population[it]=new_population[it];
    }
    new_population.erase(new_population.begin(),new_population.end());
  }
  State result = population[0];
  for(int it = 0;it < population.size();it++) {
    if (result.cost > population[it].cost) {
      result = population[it];
    }
  }
  return result;
}
