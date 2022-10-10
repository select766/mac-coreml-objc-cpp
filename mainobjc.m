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

void run_model_once_prediction(DlShogiResnet15x224SwishBatch* model, float *input_data, int batch_size, int verify, float* output_policy_expected, float* output_value_expected) {
    NSError *error = nil;

    MLMultiArray *model_input = [[MLMultiArray alloc] initWithDataPointer:input_data shape:@[[NSNumber numberWithInt:batch_size], @119, @9, @9] dataType:MLMultiArrayDataTypeFloat32 strides:@[@(119*9*9), @(9*9), @9, @1] deallocator:NULL error:NULL];

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


void run_model_once_predictions(DlShogiResnet15x224SwishBatch* model, float *input_data, int batch_size, int verify, float* output_policy_expected, float* output_value_expected) {
    NSError *error = nil;

    NSMutableArray<DlShogiResnet15x224SwishBatchInput*> *model_inputs = [NSMutableArray arrayWithCapacity:(NSUInteger)batch_size];
    for (int i = 0; i < batch_size; i++) {
        MLMultiArray *model_input = [[MLMultiArray alloc] initWithDataPointer:&input_data[i*119*9*9] shape:@[@1, @119, @9, @9] dataType:MLMultiArrayDataTypeFloat32 strides:@[@(119*9*9), @(9*9), @9, @1] deallocator:NULL error:NULL];
        DlShogiResnet15x224SwishBatchInput *input_ = [[DlShogiResnet15x224SwishBatchInput alloc] initWithInput:model_input];
        [model_inputs addObject:input_];
    }

    MLPredictionOptions *options = [[MLPredictionOptions alloc] init];

    NSArray<DlShogiResnet15x224SwishBatchOutput *> *model_outputs = [model predictionsFromInputs:model_inputs options:options error:&error];
    if (error) {
        NSLog(@"%@", error);
        exit(1);
    }

    if (verify) {
        for (int i = 0; i < batch_size; i++) {
            DlShogiResnet15x224SwishBatchOutput *model_output = model_outputs[i];
            MLMultiArray *output_policy = model_output.output_policy;
            MLMultiArray *output_value = model_output.output_value;
            // NSLog(@"%@", output_policy);
            // NSLog(@"%@", output_value);

            fprintf(stderr, "Comparing output_policy to test case\n");
            check_result(&output_policy_expected[i*policy_size], (float*)output_policy.dataPointer, policy_size);
            fprintf(stderr, "Comparing output_value to test case\n");
            check_result(&output_value_expected[i*value_size], (float*)output_value.dataPointer, value_size);
        }
    }
}

void run_model_once(DlShogiResnet15x224SwishBatch* model, float *input_data, int batch_size, int verify, float* output_policy_expected, float* output_value_expected, bool use_predictions) {
    if (use_predictions) {
        run_model_once_predictions(model, input_data, batch_size, verify, output_policy_expected, output_value_expected);
    } else {
        run_model_once_prediction(model, input_data, batch_size, verify, output_policy_expected, output_value_expected);
    }
}

long long get_ns() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

int main(int argc, const char** argv) {
    if (argc < 4) {
        fprintf(stderr, "mainobjc batch_size backend run_time_sec [use_predictions]\n"
        "backend: all, cpuandgpu, cpuonly\n"
        "example: ./mainobjc 16 all 60 0\n"
        );
        exit(1);
    }
    const int batch_size = atoi(argv[1]);
    const double run_time = atof(argv[3]);
    const char* backend = argv[2];
    const bool use_predictions = argc >= 5 ? atoi(argv[4]) != 0 : false;

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

    run_model_once(model, input_data, batch_size, 1, output_policy_expected, output_value_expected, use_predictions);

    long long start_time = get_ns();
    long long elapsed_ns = 0;
    int run_count = 0;
    int last_run_count = 0;
    long long report_cycle = 10 * 1000000000LL;
    long long next_report_elapsed = report_cycle;
    long long last_report_elapsed = 0;
    while (1) {
        elapsed_ns = get_ns() - start_time;
        if (elapsed_ns > run_time * 1000000000LL) {
            break;
        }
        if (elapsed_ns >= next_report_elapsed) {
            double elapsed_sec = (double)(elapsed_ns - last_report_elapsed) / 1000000000.0;
            double inference_per_sec = (double)(run_count - last_run_count) / elapsed_sec;
            double sample_per_sec = inference_per_sec * batch_size;
            printf("%f sec elapsed: %f inference / sec, %f samples / sec from last report\n", (double)elapsed_ns / 1000000000.0, inference_per_sec, sample_per_sec);
            last_report_elapsed = elapsed_ns;
            last_run_count = run_count;
            next_report_elapsed += report_cycle;
        }
        run_model_once(model, input_data, batch_size, 0, output_policy_expected, output_value_expected, use_predictions);
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
