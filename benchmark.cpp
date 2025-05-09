#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <random>
#include <algorithm>
#include <iomanip>
#include <omp.h>
#include <thread>
#include <functional>
#include "TextProcessor.h"
#include "transformerModel.cpp"

// Include the transformer model header
// (for actual implementation, place the transformer code in a separate header file)

using namespace std;

// Simple function to check if a file exists
bool fileExists(const string& filename) {
    ifstream file(filename);
    return file.good();
}

// Load text file
string loadTextFile(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return "";
    }
    
    stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// Benchmark a single approach with timing
template<typename Func>
double benchmark(const string& name, int num_threads, Func func) {
    cout << "Running " << name << " with " << num_threads << " threads..." << endl;
    
    auto start = chrono::high_resolution_clock::now();
    func();
    auto end = chrono::high_resolution_clock::now();
    
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    double seconds = duration.count() / 1000.0;
    
    cout << "  Completed in " << fixed << setprecision(4) << seconds << " seconds" << endl;
    return seconds;
}

// Run comprehensive benchmarks
void runAllBenchmarks(TransformerModel& model, TextProcessor& processor, const string& text) {
    cout << "\n====== PARALLELIZATION BENCHMARKS ======\n";
    
    // Create test sequences of increasing length
    vector<vector<int>> testSequences;
    vector<int> sequenceLengths = {8, 16};
    
    for (int length : sequenceLengths) {
        // Create 10 random sequences of this length
        for (int i = 0; i < 10; i++) {
            string textSlice = text.substr(i * 100, length * 10); // Different starting points
            vector<string> tokens = processor.tokenize(textSlice);
            
            if (tokens.size() > length) {
                tokens.resize(length);
                vector<int> ids = processor.words_to_ids(tokens);
                testSequences.push_back(ids);
            }
        }
    }
    
    cout << "Created " << testSequences.size() << " test sequences" << endl;
    
    // Benchmark different threading configurations
    vector<int> threadCounts = {1, 2, 4, 8};
    int maxThreads = thread::hardware_concurrency();
    
    if (maxThreads > 8) {
        threadCounts.push_back(16);
    }
    
    cout << "Hardware concurrency: " << maxThreads << " threads" << endl;
    
    // Results storage
    vector<vector<double>> results;
    vector<string> methodNames = {"Sequential", "OpenMP", "Thread-based"};
    
    // Sequential baseline (1 thread)
    vector<double> seqTimes;
    for (int length : sequenceLengths) {
        // Filter sequences of this length
        vector<vector<int>> seqs;
        for (const auto& seq : testSequences) {
            if (seq.size() == length) {
                seqs.push_back(seq);
            }
        }
        
        if (seqs.empty()) continue;
        
        double time = benchmark("Sequential (length " + to_string(length) + ")", 1, 
            [&model, &seqs]() {
                for (const auto& seq : seqs) {
                    model.forward(seq);
                }
            });
        
        seqTimes.push_back(time);
    }
    
    results.push_back(seqTimes);
    
    // OpenMP parallelism
    for (int threads : threadCounts) {
        if (threads == 1) continue; // Skip 1 thread as it's already covered by sequential
        
        vector<double> times;
        for (int length : sequenceLengths) {
            // Filter sequences of this length
            vector<vector<int>> seqs;
            for (const auto& seq : testSequences) {
                if (seq.size() == length) {
                    seqs.push_back(seq);
                }
            }
            
            if (seqs.empty()) continue;
            
            double time = benchmark("OpenMP (length " + to_string(length) + ")", threads, 
                [&model, &seqs, threads]() {
                    for (const auto& seq : seqs) {
                        model.forward_parallel(seq, threads);
                    }
                });
            
            times.push_back(time);
        }
        
        results.push_back(times);
        methodNames.push_back("OpenMP (" + to_string(threads) + " threads)");
    }
    
    // Thread-based parallelism
    for (int threads : threadCounts) {
        if (threads == 1) continue; // Skip 1 thread
        
        vector<double> times;
        for (int length : sequenceLengths) {
            // Filter sequences of this length
            vector<vector<int>> seqs;
            for (const auto& seq : testSequences) {
                if (seq.size() == length) {
                    seqs.push_back(seq);
                }
            }
            
            if (seqs.empty()) continue;
            
            double time = benchmark("Threads (length " + to_string(length) + ")", threads, 
                [&model, &seqs, threads]() {
                    for (const auto& seq : seqs) {
                        model.forward_with_threads(seq, threads);
                    }
                });
            
            times.push_back(time);
        }
        
        results.push_back(times);
        methodNames.push_back("Threads (" + to_string(threads) + " threads)");
    }
    
    // Print speedup results
    cout << "\n====== SPEEDUP SUMMARY ======\n";
    cout << "| Method | ";
    for (int length : sequenceLengths) {
        cout << "Length " << length << " | ";
    }
    cout << "Average |\n";
    cout << "|--------|";
    for (size_t i = 0; i < sequenceLengths.size(); i++) {
        cout << "---------|";
    }
    cout << "---------|\n";
    
    // Calculate and print speedups relative to sequential
    for (size_t i = 0; i < results.size(); i++) {
        cout << "| " << left << setw(6) << methodNames[i] << " | ";
        
        double sum_speedup = 0.0;
        int count = 0;
        
        for (size_t j = 0; j < results[i].size(); j++) {
            double speedup = seqTimes[j] / results[i][j];
            sum_speedup += speedup;
            count++;
            
            cout << fixed << setprecision(2) << speedup << "x | ";
        }
        
        double avg_speedup = count > 0 ? sum_speedup / count : 0.0;
        cout << fixed << setprecision(2) << avg_speedup << "x |\n";
    }
}

// Main function to showcase the transformer with attention
int main() {
    cout << "=== Transformer Model with Attention Mechanism ===" << endl;
    
    // Check for Shakespeare dataset
    string filename = "input.txt";
    if (!fileExists(filename)) {
        cerr << "Error: Could not find Shakespeare dataset at " << filename << endl;
        cerr << "Please download it from: https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt" << endl;
        return 1;
    }
    
    // Load Shakespeare text
    string text = loadTextFile(filename);
    
    if (text.empty()) {
        cerr << "Failed to load text from " << filename << endl;
        return 1;
    }
    
    cout << "Loaded Shakespeare text (" << text.size() << " characters)" << endl;
    
    // Process text
    TextProcessor processor;
    processor.build_vocab(text);
    int vocab_size = processor.get_vocab_size();
    
    // Model hyperparameters - reduced size for faster benchmarking
    int d_model = 64;      // Embedding dimension
    int num_heads = 4;     // Number of attention heads
    int d_ff = 128;        // Feed-forward hidden dimension
    int num_layers = 2;    // Number of encoder layers
    
    // Create model
    cout << "Creating transformer model with parameters:" << endl;
    cout << "  - Vocabulary size: " << vocab_size << endl;
    cout << "  - Embedding dimension: " << d_model << endl;
    cout << "  - Attention heads: " << num_heads << endl;
    cout << "  - Feed-forward dimension: " << d_ff << endl;
    cout << "  - Encoder layers: " << num_layers << endl;
    
    TransformerModel model(vocab_size, d_model, num_heads, d_ff, num_layers);
    
    // Run benchmarks
    runAllBenchmarks(model, processor, text);
    
    // Generate sample prediction
    cout << "\n====== SAMPLE PREDICTION ======\n";
    
    // Sample a random starting point in the text
    mt19937 rng(chrono::high_resolution_clock::now().time_since_epoch().count());
    uniform_int_distribution<int> dist(0, text.size() - 100);
    int start_pos = dist(rng);
    
    string sample_text = text.substr(start_pos, 50);
    vector<string> tokens = processor.tokenize(sample_text);
    
    if (tokens.size() > 10) {
        tokens.resize(10); // Use first 10 tokens as context
        
        cout << "Context: ";
        for (const auto& token : tokens) {
            cout << token << " ";
        }
        cout << endl;
        
        vector<int> token_ids = processor.words_to_ids(tokens);
        vector<double> next_token_probs = model.forward(token_ids);
        
        // Find top 5 most likely next tokens
        vector<pair<double, int>> probs_with_ids;
        for (int i = 0; i < vocab_size; i++) {
            probs_with_ids.push_back({next_token_probs[i], i});
        }
        
        // Sort by probability (descending)
        partial_sort(probs_with_ids.begin(), probs_with_ids.begin() + 5, probs_with_ids.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });
        
        cout << "Most likely next words:\n";
        
        // Convert top token IDs to words and print with probabilities
        vector<int> top_ids;
        for (int i = 0; i < 5 && i < probs_with_ids.size(); i++) {
            int token_id = probs_with_ids[i].second;
            double prob = probs_with_ids[i].first;
            top_ids.push_back(token_id);
            
            vector<string> words = processor.ids_to_words({token_id});
            cout << "  " << (i+1) << ". " << words[0] << " (";
            
            // Handle NaN values in output
            if (std::isnan(prob)) {
                cout << "N/A";
            } else {
                cout << fixed << setprecision(4) << prob * 100 << "%";
            }
            cout << ")" << endl;
        }
        
        // Generate a short sentence using top tokens
        vector<int> generation = token_ids;
        cout << "\nGenerated text: ";
        for (const auto& token : tokens) {
            cout << token << " ";
        }
        
        // Generate 20 more tokens
        for (int i = 0; i < 20; i++) {
            // Get the last n tokens (up to 10)
            vector<int> context;
            if (generation.size() <= 10) {
                context = generation;
            } else {
                context.assign(generation.end() - 10, generation.end());
            }
            
            // Predict next token
            int next_id = model.sample_next_token(context);
            generation.push_back(next_id);
            
            // Output word
            vector<string> next_word = processor.ids_to_words({next_id});
            cout << next_word[0] << " ";
        }
        cout << endl;
    }
    
   
    // Simple analytic comparison of the computational complexity
    int seq_length = 64;
    int ffnn_hidden = 256;
    
    
    
    // FFNN complexity (sequential processing)
    long ffnn_ops = seq_length * (d_model * ffnn_hidden + ffnn_hidden * d_model);
  
    
    // Transformer complexity (parallel)
    long transformer_ops = d_model * d_model * 3 + seq_length * seq_length;
    
    
    
    return 0;
}