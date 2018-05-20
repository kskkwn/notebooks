#include <iostream>
#include <string>
#include <vector>

std::vector<std::string> SufiixArray(std::string s){
  std::vector<std::string> sa;
  for (int i=0; i<s.size(); ++i) {
    sa.push_back(s.substr(i,s.size()));
  }
  return sa;
}


int main(void)
{

  std::string s=("abracadabra$");
  std::vector<std::string>sa = SufiixArray(s);

  for (int i=0; i<s.size(); ++i) {
    std::cout << sa[i] << std::endl;
  }
  return 0;
}
