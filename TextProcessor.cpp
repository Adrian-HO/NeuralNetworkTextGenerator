#include "TextProcessor.h"

TextProcessor::TextProcessor() : vocab_size(0) {}

// Tokenize text into words
vector<string> TextProcessor::tokenize(const string& text) {
    vector<string> tokens;
    string word;
    
    for (char c : text) {
        if (isalpha(c) || c == '\'') {
            word += tolower(c);
        } else if (!word.empty()) {
            tokens.push_back(word);
            word.clear();
            
            if (!isspace(c)) {
                tokens.push_back(string(1, c));
            }
        } else if (!isspace(c)) {
            tokens.push_back(string(1, c));
        }
    }
    
    if (!word.empty()) {
        tokens.push_back(word);
    }
    
    return tokens;
}

// Build vocabulary from text
void TextProcessor::build_vocab(const string& text) {
    vector<string> tokens = tokenize(text);
    unordered_map<string, int> word_counts;
    
    // Count word frequencies
    for (const auto& token : tokens) {
        word_counts[token]++;
    }
    
    // Create vocabulary (sorted by frequency)
    vector<pair<string, int>> word_freq(word_counts.begin(), word_counts.end());
    sort(word_freq.begin(), word_freq.end(), 
         [](const pair<string, int>& a, const pair<string, int>& b) {
             return a.second > b.second;
         });
    
    // Assign IDs
    word_to_id.clear();
    id_to_word.clear();
    
    for (const auto& pair : word_freq) {
        int id = word_to_id.size();
        word_to_id[pair.first] = id;
        id_to_word[id] = pair.first;
    }
    
    vocab_size = word_to_id.size();
    cout << "Vocabulary size: " << vocab_size << endl;
}

// Convert words to token IDs
vector<int> TextProcessor::words_to_ids(const vector<string>& words) {
    vector<int> ids;
    for (const auto& word : words) {
        if (word_to_id.find(word) != word_to_id.end()) {
            ids.push_back(word_to_id[word]);
        } else {
            ids.push_back(0);  // UNK token
        }
    }
    return ids;
}

// Convert token IDs to words
vector<string> TextProcessor::ids_to_words(const vector<int>& ids) {
    vector<string> words;
    for (int id : ids) {
        if (id_to_word.find(id) != id_to_word.end()) {
            words.push_back(id_to_word[id]);
        } else {
            words.push_back("<UNK>");
        }
    }
    return words;
}

int TextProcessor::get_vocab_size() const {
    return vocab_size;
}