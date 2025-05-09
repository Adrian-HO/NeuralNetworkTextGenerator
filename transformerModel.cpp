#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <unordered_map>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <iomanip>
#include <omp.h>
#include "TextProcessor.h"


using namespace std;

// Structure to hold a token and its embedding
struct Token {
    int id;
    vector<double> embedding;
};

// Structure for attention head
struct AttentionHead {
    vector<vector<double>> W_query;
    vector<vector<double>> W_key;
    vector<vector<double>> W_value;
    
    int d_model;
    int d_k;
    int d_v;
    
    // Constructor
    AttentionHead(int d_model, int d_k, int d_v, mt19937& rng) {
        this->d_model = d_model;
        this->d_k = d_k;
        this->d_v = d_v;
        
        // Initialize weights with Xavier/Glorot initialization
        uniform_real_distribution<double> dist(-sqrt(6.0 / (d_model + d_k)), sqrt(6.0 / (d_model + d_k)));
        
        // Query weights
        W_query.resize(d_k);
        for (auto& row : W_query) {
            row.resize(d_model);
            for (auto& val : row) {
                val = dist(rng);
            }
        }
        
        // Key weights
        W_key.resize(d_k);
        for (auto& row : W_key) {
            row.resize(d_model);
            for (auto& val : row) {
                val = dist(rng);
            }
        }
        
        // Value weights
        W_value.resize(d_v);
        for (auto& row : W_value) {
            row.resize(d_model);
            for (auto& val : row) {
                val = dist(rng);
            }
        }
    }
    
    // Matrix multiplication: A(m,n) * B(n,p) = C(m,p)
    vector<vector<double>> matmul(const vector<vector<double>>& A, const vector<vector<double>>& B) {
        size_t m = A.size();
        size_t n = A[0].size();
        size_t p = B[0].size();
        
        vector<vector<double>> C(m, vector<double>(p, 0.0));
        
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < p; j++) {
                for (size_t k = 0; k < n; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return C;
    }
    
    // Matrix multiplication: A(m,n) * B(n,1) = C(m,1)
    vector<double> matmul(const vector<vector<double>>& A, const vector<double>& B) {
        size_t m = A.size();
        size_t n = A[0].size();
        
        vector<double> C(m, 0.0);
        
        for (size_t i = 0; i < m; i++) {
            for (size_t k = 0; k < n; k++) {
                C[i] += A[i][k] * B[k];
            }
        }
        return C;
    }
    
    // Matrix transpose
    vector<vector<double>> transpose(const vector<vector<double>>& A) {
        size_t m = A.size();
        size_t n = A[0].size();
        
        vector<vector<double>> AT(n, vector<double>(m, 0.0));
        
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < n; j++) {
                AT[j][i] = A[i][j];
            }
        }
        return AT;
    }
    
    // Dot product of two vectors
    double dot(const vector<double>& a, const vector<double>& b) {
        double result = 0.0;
        for (size_t i = 0; i < a.size(); i++) {
            result += a[i] * b[i];
        }
        return result;
    }
    
    // Softmax function
    vector<double> softmax(const vector<double>& x) {
        vector<double> result(x.size());
    
        // Check if input contains NaN values
        for (double val : x) {
            if (std::isnan(val)) {
                // Return uniform distribution if input contains NaN
                for (size_t i = 0; i < x.size(); i++) {
                    result[i] = 1.0 / x.size();
                }
                return result;
            }
        }
        
        // Find maximum value for numerical stability
        double max_val = *max_element(x.begin(), x.end());
        
        // Compute exp(x - max_val) for numerical stability
        double sum = 0.0;
        for (size_t i = 0; i < x.size(); i++) {
            result[i] = exp(x[i] - max_val);
            sum += result[i];
        }
        
        // Check for zero sum
        if (sum < 1e-10) {
            // Return uniform distribution if sum is too small
            for (size_t i = 0; i < x.size(); i++) {
                result[i] = 1.0 / x.size();
            }
        } else {
            // Normalize
            for (size_t i = 0; i < x.size(); i++) {
                result[i] /= sum;
            }
        }
        
        return result;
    }
    
    // Compute attention scores and context vectors
    vector<vector<double>> forward(const vector<vector<double>>& input_embeddings) {
        // Convert input embeddings to query, key, and value matrices
        vector<vector<double>> Q = matmul(W_query, transpose(input_embeddings));
        vector<vector<double>> K = matmul(W_key, transpose(input_embeddings));
        vector<vector<double>> V = matmul(W_value, transpose(input_embeddings));
        
        // Compute attention scores (Q * K^T)
        vector<vector<double>> scores = matmul(transpose(Q), K);
        
        // Scale by sqrt(d_k)
        double scale = sqrt(d_k);
        for (auto& row : scores) {
            for (auto& val : row) {
                val /= scale;
            }
        }
        
        // Apply softmax to each row
        vector<vector<double>> attention_weights(scores.size());
        for (size_t i = 0; i < scores.size(); i++) {
            attention_weights[i] = softmax(scores[i]);
        }
        
        // Compute weighted sum to get context vectors
        vector<vector<double>> context = matmul(attention_weights, transpose(V));
        
        return transpose(context);
    }
    
    // Parallel forward pass
    vector<vector<double>> forward_parallel(const vector<vector<double>>& input_embeddings, int num_threads) {
        // Convert input embeddings to query, key, and value matrices
        vector<vector<double>> QT = transpose(input_embeddings);
        vector<vector<double>> Q(d_k, vector<double>(QT.size(), 0.0));
        vector<vector<double>> K(d_k, vector<double>(QT.size(), 0.0));
        vector<vector<double>> V(d_v, vector<double>(QT.size(), 0.0));
        
        // Parallelize the matrix multiplications
        #pragma omp parallel num_threads(num_threads)
        {
            // Compute Q, K, V in parallel
            #pragma omp for
            for (size_t i = 0; i < d_k; i++) {
                for (size_t j = 0; j < QT.size(); j++) {
                    for (size_t k = 0; k < d_model; k++) {
                        Q[i][j] += W_query[i][k] * QT[j][k];
                        K[i][j] += W_key[i][k] * QT[j][k];
                    }
                }
            }
            
            #pragma omp for
            for (size_t i = 0; i < d_v; i++) {
                for (size_t j = 0; j < QT.size(); j++) {
                    for (size_t k = 0; k < d_model; k++) {
                        V[i][j] += W_value[i][k] * QT[j][k];
                    }
                }
            }
        }
        
        // Compute attention scores (Q^T * K)
        vector<vector<double>> QTrans = transpose(Q);
        vector<vector<double>> scores(QTrans.size(), vector<double>(K[0].size(), 0.0));
        
        #pragma omp parallel for num_threads(num_threads)
        for (size_t i = 0; i < QTrans.size(); i++) {
            for (size_t j = 0; j < K[0].size(); j++) {
                for (size_t k = 0; k < d_k; k++) {
                    scores[i][j] += QTrans[i][k] * K[k][j];
                }
                // Scale by sqrt(d_k)
                scores[i][j] /= sqrt(d_k);
            }
        }
        
        // Apply softmax to each row
        vector<vector<double>> attention_weights(scores.size(), vector<double>(scores[0].size()));
        
        #pragma omp parallel for num_threads(num_threads)
        for (size_t i = 0; i < scores.size(); i++) {
            attention_weights[i] = softmax(scores[i]);
        }
        
        // Compute weighted sum to get context vectors
        vector<vector<double>> VTrans = transpose(V);
        vector<vector<double>> context(attention_weights.size(), vector<double>(VTrans[0].size(), 0.0));
        
        #pragma omp parallel for num_threads(num_threads)
        for (size_t i = 0; i < attention_weights.size(); i++) {
            for (size_t j = 0; j < VTrans[0].size(); j++) {
                for (size_t k = 0; k < attention_weights[0].size(); k++) {
                    context[i][j] += attention_weights[i][k] * VTrans[k][j];
                }
            }
        }
        
        return context;
    }
};

// Multi-head Attention
class MultiHeadAttention {
private:
    int num_heads;
    int d_model;
    int d_k;
    int d_v;
    int d_output;
    vector<AttentionHead> heads;
    vector<vector<double>> W_output; // Linear projection after concatenation
    mt19937 rng;
    
public:
    MultiHeadAttention(int num_heads, int d_model, int d_k, int d_v, int d_output, unsigned seed = 123) {
        this->num_heads = num_heads;
        this->d_model = d_model;
        this->d_k = d_k;
        this->d_v = d_v;
        this->d_output = d_output;
        
        // Initialize random number generator
        rng = mt19937(seed);
        
        // Initialize attention heads
        for (int i = 0; i < num_heads; i++) {
            heads.emplace_back(d_model, d_k, d_v, rng);
        }
        
        // Initialize output weights
        uniform_real_distribution<double> dist(-sqrt(6.0 / (num_heads * d_v + d_output)), 
                                              sqrt(6.0 / (num_heads * d_v + d_output)));
        
        W_output.resize(d_output);
        for (auto& row : W_output) {
            row.resize(num_heads * d_v);
            for (auto& val : row) {
                val = dist(rng);
            }
        }
    }
    
    // Sequential forward pass
    vector<vector<double>> forward(const vector<vector<double>>& input_embeddings) {
        // Compute attention for each head
        vector<vector<vector<double>>> head_outputs;
        for (int i = 0; i < num_heads; i++) {
            head_outputs.push_back(heads[i].forward(input_embeddings));
        }
        
        // Concatenate outputs from each head
        size_t seq_len = head_outputs[0].size();
        vector<vector<double>> concatenated(seq_len, vector<double>(num_heads * d_v));
        
        for (size_t i = 0; i < seq_len; i++) {
            size_t offset = 0;
            for (int h = 0; h < num_heads; h++) {
                for (int j = 0; j < d_v; j++) {
                    concatenated[i][offset + j] = head_outputs[h][i][j];
                }
                offset += d_v;
            }
        }
        
        // Project to output dimension
        vector<vector<double>> output(seq_len, vector<double>(d_output, 0.0));
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t j = 0; j < d_output; j++) {
                for (size_t k = 0; k < num_heads * d_v; k++) {
                    output[i][j] += concatenated[i][k] * W_output[j][k];
                }
            }
        }
        
        return output;
    }
    
    // Parallel forward pass across heads and matrix operations
    vector<vector<double>> forward_parallel(const vector<vector<double>>& input_embeddings, int num_threads) {
        // Compute attention for each head in parallel
        vector<vector<vector<double>>> head_outputs(num_heads);
        
        #pragma omp parallel num_threads(num_threads)
        {
            #pragma omp for
            for (int i = 0; i < num_heads; i++) {
                head_outputs[i] = heads[i].forward_parallel(input_embeddings, 1); // Use 1 thread per head
            }
        }
        
        // Concatenate outputs from each head
        size_t seq_len = head_outputs[0].size();
        vector<vector<double>> concatenated(seq_len, vector<double>(num_heads * d_v));
        
        #pragma omp parallel for num_threads(num_threads)
        for (size_t i = 0; i < seq_len; i++) {
            size_t offset = 0;
            for (int h = 0; h < num_heads; h++) {
                for (int j = 0; j < d_v; j++) {
                    concatenated[i][offset + j] = head_outputs[h][i][j];
                }
                offset += d_v;
            }
        }
        
        // Project to output dimension
        vector<vector<double>> output(seq_len, vector<double>(d_output, 0.0));
        
        #pragma omp parallel for num_threads(num_threads)
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t j = 0; j < d_output; j++) {
                for (size_t k = 0; k < num_heads * d_v; k++) {
                    output[i][j] += concatenated[i][k] * W_output[j][k];
                }
            }
        }
        
        return output;
    }
    
    // Using thread-based parallelism
    vector<vector<double>> forward_with_threads(const vector<vector<double>>& input_embeddings, int num_threads) {
        // Compute attention for each head using threads
        vector<vector<vector<double>>> head_outputs(num_heads);
        vector<thread> threads;
        
        // Create a lambda to process each head
        auto process_head = [&](int head_idx) {
            head_outputs[head_idx] = heads[head_idx].forward(input_embeddings);
        };
        
        // Launch threads
        for (int i = 0; i < num_heads; i++) {
            threads.emplace_back(process_head, i);
        }
        
        // Join threads
        for (auto& t : threads) {
            t.join();
        }
        
        // Rest of the processing remains the same
        // Concatenate outputs from each head
        size_t seq_len = head_outputs[0].size();
        vector<vector<double>> concatenated(seq_len, vector<double>(num_heads * d_v));
        
        for (size_t i = 0; i < seq_len; i++) {
            size_t offset = 0;
            for (int h = 0; h < num_heads; h++) {
                for (int j = 0; j < d_v; j++) {
                    concatenated[i][offset + j] = head_outputs[h][i][j];
                }
                offset += d_v;
            }
        }
        
        // Project to output dimension
        vector<vector<double>> output(seq_len, vector<double>(d_output, 0.0));
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t j = 0; j < d_output; j++) {
                for (size_t k = 0; k < num_heads * d_v; k++) {
                    output[i][j] += concatenated[i][k] * W_output[j][k];
                }
            }
        }
        
        return output;
    }
};

// Position-wise Feed Forward Network
class FeedForward {
private:
    int d_model;
    int d_ff;
    vector<vector<double>> W1;
    vector<vector<double>> W2;
    vector<double> b1;
    vector<double> b2;
    mt19937 rng;
    
public:
    FeedForward(int d_model, int d_ff, unsigned seed = 123) {
        this->d_model = d_model;
        this->d_ff = d_ff;
        
        // Initialize random number generator
        rng = mt19937(seed);
        
        // Initialize weights with Xavier/Glorot initialization
        uniform_real_distribution<double> dist1(-sqrt(6.0 / (d_model + d_ff)), sqrt(6.0 / (d_model + d_ff)));
        uniform_real_distribution<double> dist2(-sqrt(6.0 / (d_ff + d_model)), sqrt(6.0 / (d_ff + d_model)));
        
        // First layer
        W1.resize(d_ff);
        for (auto& row : W1) {
            row.resize(d_model);
            for (auto& val : row) {
                val = dist1(rng);
            }
        }
        
        b1.resize(d_ff, 0.0);
        
        // Second layer
        W2.resize(d_model);
        for (auto& row : W2) {
            row.resize(d_ff);
            for (auto& val : row) {
                val = dist2(rng);
            }
        }
        
        b2.resize(d_model, 0.0);
    }
    
    // ReLU activation function
    double relu(double x) {
        return max(0.0, x);
    }
    
    // Matrix multiplication with bias and activation
    vector<vector<double>> forward(const vector<vector<double>>& input) {
        size_t seq_len = input.size();
        
        // First layer: input * W1 + b1, followed by ReLU
        vector<vector<double>> hidden(seq_len, vector<double>(d_ff, 0.0));
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t j = 0; j < d_ff; j++) {
                double sum = b1[j];
                for (size_t k = 0; k < d_model; k++) {
                    sum += input[i][k] * W1[j][k];
                }
                hidden[i][j] = relu(sum);
            }
        }
        
        // Second layer: hidden * W2 + b2
        vector<vector<double>> output(seq_len, vector<double>(d_model, 0.0));
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t j = 0; j < d_model; j++) {
                double sum = b2[j];
                for (size_t k = 0; k < d_ff; k++) {
                    sum += hidden[i][k] * W2[j][k];
                }
                output[i][j] = sum;
            }
        }
        
        return output;
    }
    
    // Parallel forward pass
    vector<vector<double>> forward_parallel(const vector<vector<double>>& input, int num_threads) {
        size_t seq_len = input.size();
        
        // First layer: input * W1 + b1, followed by ReLU
        vector<vector<double>> hidden(seq_len, vector<double>(d_ff, 0.0));
        
        #pragma omp parallel for num_threads(num_threads)
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t j = 0; j < d_ff; j++) {
                double sum = b1[j];
                for (size_t k = 0; k < d_model; k++) {
                    sum += input[i][k] * W1[j][k];
                }
                hidden[i][j] = relu(sum);
            }
        }
        
        // Second layer: hidden * W2 + b2
        vector<vector<double>> output(seq_len, vector<double>(d_model, 0.0));
        
        #pragma omp parallel for num_threads(num_threads)
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t j = 0; j < d_model; j++) {
                double sum = b2[j];
                for (size_t k = 0; k < d_ff; k++) {
                    sum += hidden[i][k] * W2[j][k];
                }
                output[i][j] = sum;
            }
        }
        
        return output;
    }
};

// Layer Normalization
class LayerNorm {
private:
    int d_model;
    vector<double> gamma; // Scale parameter
    vector<double> beta;  // Shift parameter
    double eps;
    
public:
    LayerNorm(int d_model, double eps = 1e-5) {
        this->d_model = d_model;
        this->eps = eps;
        
        // Initialize gamma and beta
        gamma.resize(d_model, 1.0); // Initialize to ones
        beta.resize(d_model, 0.0);  // Initialize to zeros
    }
    
    vector<vector<double>> forward(const vector<vector<double>>& input) {
        size_t seq_len = input.size();
        vector<vector<double>> output(seq_len, vector<double>(d_model, 0.0));
        
        for (size_t i = 0; i < seq_len; i++) {
            // Compute mean
            double mean = 0.0;
            for (int j = 0; j < d_model; j++) {
                mean += input[i][j];
            }
            mean /= d_model;
            
            // Compute variance
            double var = 0.0;
            for (int j = 0; j < d_model; j++) {
                var += (input[i][j] - mean) * (input[i][j] - mean);
            }
            var /= d_model;
            
            // Normalize and apply gamma and beta
            for (int j = 0; j < d_model; j++) {
                output[i][j] = gamma[j] * (input[i][j] - mean) / sqrt(var + eps) + beta[j];
            }
        }
        
        return output;
    }
    
    vector<vector<double>> forward_parallel(const vector<vector<double>>& input, int num_threads) {
        size_t seq_len = input.size();
        vector<vector<double>> output(seq_len, vector<double>(d_model, 0.0));
        
        #pragma omp parallel for num_threads(num_threads)
        for (size_t i = 0; i < seq_len; i++) {
            // Compute mean
            double mean = 0.0;
            for (int j = 0; j < d_model; j++) {
                mean += input[i][j];
            }
            mean /= d_model;
            
            // Compute variance
            double var = 0.0;
            for (int j = 0; j < d_model; j++) {
                var += (input[i][j] - mean) * (input[i][j] - mean);
            }
            var /= d_model;
            
            // Normalize and apply gamma and beta
            for (int j = 0; j < d_model; j++) {
                output[i][j] = gamma[j] * (input[i][j] - mean) / sqrt(var + eps) + beta[j];
            }
        }
        
        return output;
    }
};

// Encoder Layer
class EncoderLayer {
private:
    MultiHeadAttention attention;
    FeedForward feed_forward;
    LayerNorm norm1;
    LayerNorm norm2;
    int d_model;
    
public:
    EncoderLayer(int num_heads, int d_model, int d_k, int d_v, int d_ff, unsigned seed = 123)
        : attention(num_heads, d_model, d_k, d_v, d_model, seed),
          feed_forward(d_model, d_ff, seed),
          norm1(d_model),
          norm2(d_model) {
        this->d_model = d_model;
    }
    
    vector<vector<double>> forward(const vector<vector<double>>& input) {
        // Self-attention
        vector<vector<double>> attention_output = attention.forward(input);
        
        // Residual connection and normalization
        vector<vector<double>> norm_output1(input.size(), vector<double>(d_model, 0.0));
        for (size_t i = 0; i < input.size(); i++) {
            for (int j = 0; j < d_model; j++) {
                norm_output1[i][j] = input[i][j] + attention_output[i][j];
            }
        }
        norm_output1 = norm1.forward(norm_output1);
        
        // Feed-forward
        vector<vector<double>> ff_output = feed_forward.forward(norm_output1);
        
        // Residual connection and normalization
        vector<vector<double>> norm_output2(input.size(), vector<double>(d_model, 0.0));
        for (size_t i = 0; i < input.size(); i++) {
            for (int j = 0; j < d_model; j++) {
                norm_output2[i][j] = norm_output1[i][j] + ff_output[i][j];
            }
        }
        return norm2.forward(norm_output2);
    }
    
    vector<vector<double>> forward_parallel(const vector<vector<double>>& input, int num_threads) {
        // Self-attention
        vector<vector<double>> attention_output = attention.forward_parallel(input, num_threads);
        
        // Residual connection
        vector<vector<double>> norm_input1(input.size(), vector<double>(d_model, 0.0));
        
        #pragma omp parallel for num_threads(num_threads)
        for (size_t i = 0; i < input.size(); i++) {
            for (int j = 0; j < d_model; j++) {
                norm_input1[i][j] = input[i][j] + attention_output[i][j];
            }
        }
        
        // Normalization
        vector<vector<double>> norm_output1 = norm1.forward_parallel(norm_input1, num_threads);
        
        // Feed-forward
        vector<vector<double>> ff_output = feed_forward.forward_parallel(norm_output1, num_threads);
        
        // Residual connection
        vector<vector<double>> norm_input2(input.size(), vector<double>(d_model, 0.0));
        
        #pragma omp parallel for num_threads(num_threads)
        for (size_t i = 0; i < input.size(); i++) {
            for (int j = 0; j < d_model; j++) {
                norm_input2[i][j] = norm_output1[i][j] + ff_output[i][j];
            }
        }
        
        // Normalization
        return norm2.forward_parallel(norm_input2, num_threads);
    }
    
    // Using thread-based parallelism
    vector<vector<double>> forward_with_threads(const vector<vector<double>>& input, int num_threads) {
        // Self-attention with threads
        vector<vector<double>> attention_output = attention.forward_with_threads(input, num_threads);
        
        // Rest of processing remains sequential
        // Residual connection and normalization
        vector<vector<double>> norm_output1(input.size(), vector<double>(d_model, 0.0));
        for (size_t i = 0; i < input.size(); i++) {
            for (int j = 0; j < d_model; j++) {
                norm_output1[i][j] = input[i][j] + attention_output[i][j];
            }
        }
        norm_output1 = norm1.forward(norm_output1);
        
        // Feed-forward
        vector<vector<double>> ff_output = feed_forward.forward(norm_output1);
        
        // Residual connection and normalization
        vector<vector<double>> norm_output2(input.size(), vector<double>(d_model, 0.0));
        for (size_t i = 0; i < input.size(); i++) {
            for (int j = 0; j < d_model; j++) {
                norm_output2[i][j] = norm_output1[i][j] + ff_output[i][j];
            }
        }
        return norm2.forward(norm_output2);
    }
};

// Transformer Model for next word prediction
class TransformerModel {
private:
    int vocab_size;
    int d_model;
    int num_layers;
    vector<vector<double>> token_embeddings;
    vector<vector<double>> position_embeddings;
    vector<EncoderLayer> encoder_layers;
    vector<vector<double>> output_weights;
    mt19937 rng;
    
public:
    TransformerModel(int vocab_size, int d_model, int num_heads, int d_ff, int num_layers, unsigned seed = 123) {
        this->vocab_size = vocab_size;
        this->d_model = d_model;
        this->num_layers = num_layers;
        
        // Initialize random number generator
        rng = mt19937(seed);
        
        // Initialize token embeddings
        uniform_real_distribution<double> dist(-0.1, 0.1);
        token_embeddings.resize(vocab_size, vector<double>(d_model));
        for (auto& embedding : token_embeddings) {
            for (auto& val : embedding) {
                val = dist(rng);
            }
        }
        
        // Initialize positional embeddings (fixed sinusoidal)
        position_embeddings.resize(1000, vector<double>(d_model, 0.0)); // Support sequences up to length 1000
        for (int pos = 0; pos < 1000; pos++) {
            for (int i = 0; i < d_model; i += 2) {
                position_embeddings[pos][i] = sin(pos / pow(10000, (2 * i) / d_model));
                if (i + 1 < d_model) {
                    position_embeddings[pos][i + 1] = cos(pos / pow(10000, (2 * i) / d_model));
                }
            }
        }
        
        // Initialize encoder layers
        int d_k = d_model / num_heads;
        int d_v = d_model / num_heads;
        for (int i = 0; i < num_layers; i++) {
            encoder_layers.emplace_back(num_heads, d_model, d_k, d_v, d_ff, seed + i);
        }
        
        // Initialize output projection weights
        output_weights.resize(vocab_size, vector<double>(d_model, 0.0));
        for (auto& row : output_weights) {
            for (auto& val : row) {
                val = dist(rng);
            }
        }
    }
    
    // Apply embeddings to input tokens
    vector<vector<double>> embed_tokens(const vector<int>& tokens) {
        size_t seq_len = tokens.size();
        vector<vector<double>> embeddings(seq_len, vector<double>(d_model, 0.0));
        
        for (size_t i = 0; i < seq_len; i++) {
            // Add token embedding
            int token_id = tokens[i];
            for (int j = 0; j < d_model; j++) {
                embeddings[i][j] = token_embeddings[token_id][j];
            }
            
            // Add positional embedding
            for (int j = 0; j < d_model; j++) {
                embeddings[i][j] += position_embeddings[i][j];
            }
        }
        
        return embeddings;
    }
    
    // Softmax function
    vector<double> softmax(const vector<double>& x) {
        vector<double> result(x.size());
        double max_val = *max_element(x.begin(), x.end());
        double sum = 0.0;
        
        for (size_t i = 0; i < x.size(); i++) {
            result[i] = exp(x[i] - max_val); // Shifted for numerical stability
            sum += result[i];
        }
        
        for (size_t i = 0; i < x.size(); i++) {
            result[i] /= sum;
        }
        
        return result;
    }
    
    // Forward pass through the model
    vector<double> forward(const vector<int>& tokens) {
         // Embed input tokens
    vector<vector<double>> embeddings = embed_tokens(tokens);
    
    // Pass through encoder layers
    vector<vector<double>> encoder_output = embeddings;
    for (int i = 0; i < num_layers; i++) {
        encoder_output = encoder_layers[i].forward(encoder_output);
        
        // Check for NaN values after each layer
        for (auto& row : encoder_output) {
            for (auto& val : row) {
                if (std::isnan(val)) {
                    val = 0.0; // Replace NaN with 0
                }
            }
        }
    }
    
    // Project to vocabulary space (using the last token's output)
    vector<double> logits(vocab_size, 0.0);
    for (int i = 0; i < vocab_size; i++) {
        for (int j = 0; j < d_model; j++) {
            double product = encoder_output.back()[j] * output_weights[i][j];
            // Check for NaN in the product
            if (!std::isnan(product)) {
                logits[i] += product;
            }
        }
        
        // Check for NaN in logits
        if (std::isnan(logits[i])) {
            logits[i] = 0.0;
        }
    }
    
    // Apply softmax to get probabilities
    return softmax(logits);
    }
    
    // Parallel forward pass
    vector<double> forward_parallel(const vector<int>& tokens, int num_threads) {
        // Embed input tokens
        vector<vector<double>> embeddings = embed_tokens(tokens);
        
        // Pass through encoder layers in parallel
        vector<vector<double>> encoder_output = embeddings;
        for (int i = 0; i < num_layers; i++) {
            encoder_output = encoder_layers[i].forward_parallel(encoder_output, num_threads);
        }
        
        // Project to vocabulary space (using the last token's output)
        vector<double> logits(vocab_size, 0.0);
        
        #pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < vocab_size; i++) {
            for (int j = 0; j < d_model; j++) {
                logits[i] += encoder_output.back()[j] * output_weights[i][j];
            }
        }
        
        // Apply softmax to get probabilities
        return softmax(logits);
    }
    
    // Using thread-based parallelism
    vector<double> forward_with_threads(const vector<int>& tokens, int num_threads) {
        // Embed input tokens
        vector<vector<double>> embeddings = embed_tokens(tokens);
        
        // Pass through encoder layers with threads
        vector<vector<double>> encoder_output = embeddings;
        for (int i = 0; i < num_layers; i++) {
            encoder_output = encoder_layers[i].forward_with_threads(encoder_output, num_threads);
        }
        
        // Project to vocabulary space (using the last token's output)
        vector<double> logits(vocab_size, 0.0);
        for (int i = 0; i < vocab_size; i++) {
            for (int j = 0; j < d_model; j++) {
                logits[i] += encoder_output.back()[j] * output_weights[i][j];
            }
        }
        
        // Apply softmax to get probabilities
        return softmax(logits);
    }
    
    // Predict the next token
    int predict_next_token(const vector<int>& tokens) {
        vector<double> probs = forward(tokens);
        return max_element(probs.begin(), probs.end()) - probs.begin();
    }
    
    // Predict the next token with sampling
    int sample_next_token(const vector<int>& tokens) {
        vector<double> probs = forward(tokens);
        
        // Sample from the distribution
        uniform_real_distribution<double> dist(0.0, 1.0);
        double r = dist(rng);
        double cum_prob = 0.0;
        
        for (int i = 0; i < vocab_size; i++) {
            cum_prob += probs[i];
            if (r <= cum_prob) {
                return i;
            }
        }
        
        return vocab_size - 1; // Fallback
    }
};