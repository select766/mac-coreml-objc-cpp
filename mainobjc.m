#import <stdio.h>
#import <time.h>
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import "DlShogiResnet15x224SwishBatch.h"

const int test_case_size = 1024;
const int input_size = 119 * 9 * 9;
const int policy_size = 2187;
const int value_size = 1;

void malloc_read(float** dst, FILE* fp, size_t size) {
    *dst = (float*)malloc(size);
    if (!*dst) {
        fputs("failed to malloc\n", stderr);
        exit(1);
    }
    if (fread(*dst, 1, size, fp) != size) {
        fputs("failed to read test case\n", stderr);
        exit(1);
    }
}

void read_test_case(float** input, float** output_policy, float** output_value) {
    FILE *fin;
    if ((fin = fopen("SampleIO15x224MyData.bin", "rb")) == NULL) {
        fputs("failed to open test case\n", stderr);
        exit(1);
    }

    // input: [test_case_size, 119, 9, 9], output_policy: [test_case_size, 2187], output_value: [test_case_size, 1]
    // 全てfloat32でこの順で保存されている
    // output_valueはsigmoidがかかっていない

    malloc_read(input, fin, test_case_size * input_size * sizeof(float));
    malloc_read(output_policy, fin, test_case_size * policy_size * sizeof(float));
    malloc_read(output_value, fin, test_case_size * value_size * sizeof(float));
}

void check_result(const float* expected, const float* actual, int count) {
    float rtol = 1e-1, atol = 5e-2;
    float max_diff = 0.0F;
    for (int i = 0; i < count; i++) {
        float e = expected[i];
        float a = actual[i];
        float diff = fabsf(e - a);
        float tol = atol + rtol * fabsf(e);
        if (diff > tol) {
            fprintf(stderr, "Error at index %d: %f != %f\n", i, e, a);
            return;
        }
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    fprintf(stderr, "Max difference: %f\n", max_diff);
    return;
}

void run_model_once(DlShogiResnet15x224SwishBatch* model, MLMultiArray *model_input, int batch_size, int verify, float* output_policy_expected, float* output_value_expected) {
    NSError *error = nil;

    DlShogiResnet15x224SwishBatchOutput *model_output = [model predictionFromInput:model_input error:&error];
    if (error) {
        NSLog(@"%@", error);
        exit(1);
    }

    if (verify) {
        MLMultiArray *output_policy = model_output.output_policy;
        MLMultiArray *output_value = model_output.output_value;
        // NSLog(@"%@", output_policy);
        // NSLog(@"%@", output_value);

        fprintf(stderr, "Comparing output_policy to test case\n");
        check_result(output_policy_expected, (float*)output_policy.dataPointer, batch_size * policy_size);
        fprintf(stderr, "Comparing output_value to test case\n");
        check_result(output_value_expected, (float*)output_value.dataPointer, batch_size * value_size);
    }
}

long long get_ns() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

int main(int argc, const char** argv) {
    if (argc != 4) {
        fprintf(stderr, "mainobjc batch_size backend run_time_sec\n"
        "backend: all, cpuandgpu, cpuonly\n"
        "example: ./mainobjc 16 all 60\n"
        );
        exit(1);
    }
    const int batch_size = atoi(argv[1]);
    const double run_time = atof(argv[3]);
    const char* backend = argv[2];

    MLModelConfiguration* config = [MLModelConfiguration new];
    // 使用デバイス
    // MLComputeUnitsCPUOnly = 0,
    // MLComputeUnitsCPUAndGPU = 1,
    // MLComputeUnitsAll = 2
    MLComputeUnits computeUnits;
    if (strcmp(backend, "cpuonly") == 0) {
        computeUnits = MLComputeUnitsCPUOnly;
    } else if (strcmp(backend, "cpuandgpu") == 0) {
        computeUnits = MLComputeUnitsCPUAndGPU;
    } else if (strcmp(backend, "all") == 0) {
        computeUnits = MLComputeUnitsAll;
    } else {
        fprintf(stderr, "Unknown backend %s\n", backend);
        exit(1);
    }
    config.computeUnits = computeUnits;

    NSError *error = nil;
    // カレントディレクトリの DlShogiResnet15x224SwishBatch.mlmodelc が読まれる (存在しないとmlmodel==null, error!=null)
    DlShogiResnet15x224SwishBatch* model = [[DlShogiResnet15x224SwishBatch alloc] initWithConfiguration:config error:&error];
    NSLog(@"%@", model);
    NSLog(@"%@", error);

    if (!model) {
        NSLog(@"Failed to load model, %@", error);
        exit(1);
    }

    float *input_data, *output_policy_expected, *output_value_expected;
    read_test_case(&input_data, &output_policy_expected, &output_value_expected);

    MLMultiArray *model_input = [[MLMultiArray alloc] initWithDataPointer:input_data shape:@[[NSNumber numberWithInt:batch_size], @119, @9, @9] dataType:MLMultiArrayDataTypeFloat32 strides:@[@(119*9*9), @(9*9), @9, @1] deallocator:NULL error:NULL];

    run_model_once(model, model_input, batch_size, 1, output_policy_expected, output_value_expected);

    long long start_time = get_ns();
    long long elapsed_ns = 0;
    int run_count = 0;
    while (1) {
        elapsed_ns = get_ns() - start_time;
        if (elapsed_ns > run_time * 1000000000LL) {
            break;
        }
        run_model_once(model, model_input, batch_size, 0, output_policy_expected, output_value_expected);
        run_count++;
    }

    double elapsed_sec = (double)elapsed_ns / 1000000000.0;
    double inference_per_sec = (double)run_count / elapsed_sec;
    double sample_per_sec = inference_per_sec * batch_size;
    printf("Backend: %s, batch size: %d\n", backend, batch_size);
    printf("Run for %f sec\n", elapsed_sec);
    printf("%f inference / sec\n%f samples / sec\n", inference_per_sec, sample_per_sec);
    return 0;
}
