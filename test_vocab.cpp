
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <string>

int main() {
    std::ifstream f("D:\\TaoVm\\vocab.json");
    std::stringstream ss; ss << f.rdbuf();
    std::string line = ss.str();
    std::cout << "Total len: " << line.size() << std::endl;
    std::cout << "First 300 chars: " << line.substr(0, 300) << std::endl;
    return 0;
}
