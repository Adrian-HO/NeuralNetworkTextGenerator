#ifndef TEXT_PROCESSOR_H
#define TEXT_PROCESSOR_H

#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <iostream>

using namespace std;

// Text processing utilities
class TextProcessor {
private:
    unordered_map<string, int> word_to_id;
    unordered_map<int, string> id_to_word;
    int vocab_size;
    
public:
    TextProcessor();
    vector<string> tokenize(const string& text);
    void build_vocab(const string& text);
    vector<int> words_to_ids(const vector<string>& words);
    vector<string> ids_to_words(const vector<int>& ids);
    int get_vocab_size() const;
};

#endif // TEXT_PROCESSOR_H