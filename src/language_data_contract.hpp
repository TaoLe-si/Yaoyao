#pragma once
#include <vector>
#include <string>
#include <stdexcept>
namespace tao::data {
enum Special { BOS=256, USER=257, ASSISTANT=258, TURN_END=259, EOS=260 };
struct Message { bool assistant; std::string utf8; };
struct Token { int id; bool loss; };
inline std::vector<Token> encode(const std::vector<Message>&messages){
 if(messages.empty())throw std::invalid_argument("empty conversation");
 std::vector<Token>out{{BOS,false}};
 for(const auto&m:messages){out.push_back({m.assistant?ASSISTANT:USER,false});for(unsigned char c:m.utf8)out.push_back({int(c),m.assistant});out.push_back({TURN_END,m.assistant});}
 out.push_back({EOS,messages.back().assistant});return out;
}
struct Example { std::vector<int> input,target; std::vector<bool> loss; bool reset; };
inline std::vector<Example> chunks(const std::vector<Token>&t,size_t width){
 if(!width)throw std::invalid_argument("width");std::vector<Example>out;
 for(size_t begin=0;begin+1<t.size();begin+=width){Example e;e.reset=begin==0;for(size_t j=begin;j+1<t.size()&&j<begin+width;++j){e.input.push_back(t[j].id);e.target.push_back(t[j+1].id);e.loss.push_back(t[j+1].loss);}out.push_back(e);}return out;
}
}
