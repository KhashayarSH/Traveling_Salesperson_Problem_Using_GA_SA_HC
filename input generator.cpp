#include <iostream>
#include <fstream>
#include <vector>
#include <stack>
#include <set>
#include <unordered_map>
#include <string>
#include <chrono>
#include <queue>
#include <random>
std::ofstream fout("input500.txt");
double randZeroToOne()
{
    return rand() / (RAND_MAX + 1.);
}
int main()
{
    fout<<500<<'\n';
	for(int i = 0;i<500;i++){
	    fout<<randZeroToOne()<<' '<<randZeroToOne()<<'\n';
	}
	fout<<"GA\n";
	return 0;
}
