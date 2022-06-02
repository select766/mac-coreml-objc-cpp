// pure C++ code

class NNWrapper {
public:
    NNWrapper(const char* computeUnits);
    ~NNWrapper();
    bool run(int batch_size, float* input, float* output_policy, float* output_value);
    static const int input_size = 119 * 9 * 9;
    static const int policy_size = 2187;
    static const int value_size = 1;
private:
    void* model;
};
