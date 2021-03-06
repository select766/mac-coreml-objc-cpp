// 動的にmlmodelをコンパイルしてロードするサンプル

#import <stdio.h>
#import <time.h>
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

const int test_case_size = 1024;
const int input_size = 119 * 9 * 9;
const int policy_size = 2187;
const int value_size = 1;

/// Model Prediction Input Type
API_AVAILABLE(macos(10.15), ios(13.0), watchos(6.0), tvos(13.0)) __attribute__((visibility("hidden")))
@interface DlShogiResnetInput : NSObject<MLFeatureProvider>

/// input as 1 × 119 × 9 × 9 4-dimensional array of floats
@property (readwrite, nonatomic, strong) MLMultiArray * input;
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithInput:(MLMultiArray *)input NS_DESIGNATED_INITIALIZER;

@end


/// Model Prediction Output Type
API_AVAILABLE(macos(10.15), ios(13.0), watchos(6.0), tvos(13.0)) __attribute__((visibility("hidden")))
@interface DlShogiResnetOutput : NSObject<MLFeatureProvider>

/// output_policy as multidimensional array of floats
@property (readwrite, nonatomic, strong) MLMultiArray * output_policy;

/// output_value as multidimensional array of floats
@property (readwrite, nonatomic, strong) MLMultiArray * output_value;
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithOutput_policy:(MLMultiArray *)output_policy output_value:(MLMultiArray *)output_value NS_DESIGNATED_INITIALIZER;

@end

@implementation DlShogiResnetInput

- (instancetype)initWithInput:(MLMultiArray *)input {
    self = [super init];
    if (self) {
        _input = input;
    }
    return self;
}

- (NSSet<NSString *> *)featureNames {
    return [NSSet setWithArray:@[@"input"]];
}

- (nullable MLFeatureValue *)featureValueForName:(NSString *)featureName {
    if ([featureName isEqualToString:@"input"]) {
        return [MLFeatureValue featureValueWithMultiArray:_input];
    }
    return nil;
}

@end

@implementation DlShogiResnetOutput

- (instancetype)initWithOutput_policy:(MLMultiArray *)output_policy output_value:(MLMultiArray *)output_value {
    self = [super init];
    if (self) {
        _output_policy = output_policy;
        _output_value = output_value;
    }
    return self;
}

- (NSSet<NSString *> *)featureNames {
    return [NSSet setWithArray:@[@"output_policy", @"output_value"]];
}

- (nullable MLFeatureValue *)featureValueForName:(NSString *)featureName {
    if ([featureName isEqualToString:@"output_policy"]) {
        return [MLFeatureValue featureValueWithMultiArray:_output_policy];
    }
    if ([featureName isEqualToString:@"output_value"]) {
        return [MLFeatureValue featureValueWithMultiArray:_output_value];
    }
    return nil;
}

@end

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

void run_model_once(MLModel* model, float *input_data, int batch_size, int verify, float* output_policy_expected, float* output_value_expected) {
    NSError *error = nil;

    MLMultiArray *model_input = [[MLMultiArray alloc] initWithDataPointer:input_data shape:@[[NSNumber numberWithInt:batch_size], @119, @9, @9] dataType:MLMultiArrayDataTypeFloat32 strides:@[@(119*9*9), @(9*9), @9, @1] deallocator:NULL error:NULL];
    
    DlShogiResnetInput *input_ = [[DlShogiResnetInput alloc] initWithInput:model_input];
    id<MLFeatureProvider> outFeatures = [model predictionFromFeatures:input_ options:[[MLPredictionOptions alloc] init] error:&error];
    if (error) {
        NSLog(@"%@", error);
        exit(1);
    }

    DlShogiResnetOutput *model_output = [[DlShogiResnetOutput alloc] initWithOutput_policy:(MLMultiArray *)[outFeatures featureValueForName:@"output_policy"].multiArrayValue output_value:(MLMultiArray *)[outFeatures featureValueForName:@"output_value"].multiArrayValue];

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

    char* model_path = "./DlShogiResnet15x224SwishBatch.mlmodel"; // このモデルをコンパイルし、mlmodelcを生成し、それをロードする
    NSString* model_path_ns = [NSString stringWithCString: model_path encoding:NSUTF8StringEncoding];
    NSString* modelc_path_ns = [NSString stringWithFormat: @"%@c", model_path_ns];

    
    NSFileManager *filemanager = [NSFileManager defaultManager];

    NSError *error = nil;
    if (![filemanager fileExistsAtPath: modelc_path_ns]) {
        if ([filemanager fileExistsAtPath: model_path_ns]) {
            NSLog(@"Compiling model");
            NSURL* modelc_tmp_path = [MLModel compileModelAtURL: [NSURL fileURLWithPath: model_path_ns] error: &error];
            if (!modelc_tmp_path) {
                NSLog(@"Failed to compile model, %@", error);
                exit(1);
            }
            BOOL ret = [filemanager moveItemAtURL: modelc_tmp_path toURL: [NSURL fileURLWithPath: modelc_path_ns] error: &error];
            if (!ret) {
                NSLog(@"Failed to move compiled model from %@ to %@, %@", modelc_tmp_path, modelc_path_ns, error);
                exit(1);
            }
        } else {
            NSLog(@"Model %@ does not exist", model_path_ns);
            exit(1);
        }
    } else {
        NSLog(@"Loading already compiled model");
    }

    MLModel *model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath: modelc_path_ns] configuration:config error:&error];
    NSLog(@"%@", model);
    NSLog(@"%@", error);

    if (!model) {
        NSLog(@"Failed to load model, %@", error);
        exit(1);
    }

    if (![model init]) {
        NSLog(@"Failed to initialize model");
        exit(1);
    }

    float *input_data, *output_policy_expected, *output_value_expected;
    read_test_case(&input_data, &output_policy_expected, &output_value_expected);

    run_model_once(model, input_data, batch_size, 1, output_policy_expected, output_value_expected);

    long long start_time = get_ns();
    long long elapsed_ns = 0;
    int run_count = 0;
    while (1) {
        elapsed_ns = get_ns() - start_time;
        if (elapsed_ns > run_time * 1000000000LL) {
            break;
        }
        run_model_once(model, input_data, batch_size, 0, output_policy_expected, output_value_expected);
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
