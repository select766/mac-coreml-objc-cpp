// pure C++ code

#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include "nnwrapper.hpp"

using namespace std;

const int test_case_size = 1024;
const int input_size = 119 * 9 * 9;
const int policy_size = 2187;
const int value_size = 1;

void read_test_case(vector<float> &input, vector<float> &output_policy, vector<float> &output_value) {
    ifstream fin("SampleIO15x224MyData.bin", ios::in | ios::binary);
    if (!fin) {
        cerr << "failed to open test case" << endl;
        exit(1);
    }

    // input: [test_case_size, 119, 9, 9], output_policy: [test_case_size, 2187], output_value: [test_case_size, 1]
    // 全てfloat32でこの順で保存されている
    input.resize(test_case_size * input_size * sizeof(float));
    output_policy.resize(test_case_size * policy_size * sizeof(float));
    output_value.resize(test_case_size * value_size * sizeof(float));
    
    fin.read((char*)&(input[0]), test_case_size * input_size * sizeof(float));
    fin.read((char*)&(output_policy[0]), test_case_size * policy_size * sizeof(float));
    fin.read((char*)&(output_value[0]), test_case_size * value_size * sizeof(float));
    if (!fin) {
        cerr << "failed to read test case" << endl;
        exit(1);
    }
}

void check_result(const float* expected, const float* actual, int count, float rtol = 1e-1, float atol = 5e-2) {
    float max_diff = 0.0F;
    for (int i = 0; i < count; i++) {
        auto e = expected[i];
        auto a = actual[i];
        auto diff = abs(e - a);
        auto tol = atol + rtol * abs(e);
        if (diff > tol) {
            cerr << "Error at index " << i << ": " << e << " != " << a << endl;
            return;
        }
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    cerr << "Max difference: " << max_diff << endl;
}

int main(int argc, const char** argv) {
    if (argc != 4) {
        cerr << "maincpp batch_size backend run_time_sec" << endl;
        cerr << "backend: all, cpuandgpu, cpuonly" << endl;
        cerr << "example: ./maincpp 16 all 10" << endl;
        exit(1);
    }
    const int batch_size = atoi(argv[1]);
    const double run_time = atof(argv[3]);
    const char* backend = argv[2];
    vector<float> input, output_policy, output_value, output_policy_expected, output_value_expected;
    output_policy.resize(test_case_size * policy_size * sizeof(float));
    output_value.resize(test_case_size * value_size * sizeof(float));
    cerr << "reading test case" << endl;
    read_test_case(input, output_policy_expected, output_value_expected);
    cerr << "reading model" << endl;

    NNWrapper nnwrapper(backend);

    // run first time
    nnwrapper.run(batch_size, input.data(), output_policy.data(), output_value.data());
    cerr << "Comparing output_policy to test case" << endl;
    check_result(output_policy_expected.data(), output_policy.data(), batch_size * policy_size);
    cerr << "Comparing output_value to test case" << endl;
    check_result(output_value_expected.data(), output_value.data(), batch_size * value_size);

    cerr << "Benchmarking for " << run_time << " sec..." << endl;
    auto start_time = chrono::system_clock::now();
    chrono::duration<int, micro> elapsed;
    int run_count = 0;
    // run in loop
    while (1) {
        elapsed = chrono::system_clock::now() - start_time;
        if (elapsed.count() > run_time * 1000000.0) {
            break;
        }
        nnwrapper.run(batch_size, input.data(), output_policy.data(), output_value.data());

        run_count++;
    }

    double elapsed_sec = double(elapsed.count()) / 1000000.0;
    double inference_per_sec = double(run_count) / elapsed_sec;
    double sample_per_sec = inference_per_sec * batch_size;
    cout << "Backend: " << backend << ", batch size: " << batch_size << endl;
    cout << "Run for " << elapsed_sec << " sec" << endl << inference_per_sec << " inference / sec" << endl << sample_per_sec << " samples / sec" << endl;

    return 0;
}
