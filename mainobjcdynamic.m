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
@interface DlShogiResnet15x224SwishBatchInput : NSObject<MLFeatureProvider>

/// x as 1 × 119 × 9 × 9 4-dimensional array of floats
@property (readwrite, nonatomic, strong) MLMultiArray * x;
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithX:(MLMultiArray *)x NS_DESIGNATED_INITIALIZER;

@end


/// Model Prediction Output Type
API_AVAILABLE(macos(10.15), ios(13.0), watchos(6.0), tvos(13.0)) __attribute__((visibility("hidden")))
@interface DlShogiResnet15x224SwishBatchOutput : NSObject<MLFeatureProvider>

/// move as multidimensional array of floats
@property (readwrite, nonatomic, strong) MLMultiArray * move;

/// result as multidimensional array of floats
@property (readwrite, nonatomic, strong) MLMultiArray * result;
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithMove:(MLMultiArray *)move result:(MLMultiArray *)result NS_DESIGNATED_INITIALIZER;

@end


@implementation DlShogiResnet15x224SwishBatchInput

- (instancetype)initWithX:(MLMultiArray *)x {
    self = [super init];
    if (self) {
        _x = x;
    }
    return self;
}

- (NSSet<NSString *> *)featureNames {
    return [NSSet setWithArray:@[@"x"]];
}

- (nullable MLFeatureValue *)featureValueForName:(NSString *)featureName {
    if ([featureName isEqualToString:@"x"]) {
        return [MLFeatureValue featureValueWithMultiArray:_x];
    }
    return nil;
}

@end

@implementation DlShogiResnet15x224SwishBatchOutput

- (instancetype)initWithMove:(MLMultiArray *)move result:(MLMultiArray *)result {
    self = [super init];
    if (self) {
        _move = move;
        _result = result;
    }
    return self;
}

- (NSSet<NSString *> *)featureNames {
    return [NSSet setWithArray:@[@"move", @"result"]];
}

- (nullable MLFeatureValue *)featureValueForName:(NSString *)featureName {
    if ([featureName isEqualToString:@"move"]) {
        return [MLFeatureValue featureValueWithMultiArray:_move];
    }
    if ([featureName isEqualToString:@"result"]) {
        return [MLFeatureValue featureValueWithMultiArray:_result];
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

void run_model_once(MLModel* model, MLMultiArray *model_input, int batch_size, int verify, float* output_policy_expected, float* output_value_expected) {
    NSError *error = nil;
    
    DlShogiResnet15x224SwishBatchInput *input_ = [[DlShogiResnet15x224SwishBatchInput alloc] initWithX:model_input];
    id<MLFeatureProvider> outFeatures = [model predictionFromFeatures:input_ options:[[MLPredictionOptions alloc] init] error:&error];
    if (error) {
        NSLog(@"%@", error);
        exit(1);
    }

    DlShogiResnet15x224SwishBatchOutput *model_output = [[DlShogiResnet15x224SwishBatchOutput alloc] initWithMove:(MLMultiArray *)[outFeatures featureValueForName:@"move"].multiArrayValue result:(MLMultiArray *)[outFeatures featureValueForName:@"result"].multiArrayValue];

    if (verify) {
        MLMultiArray *output_move = model_output.move;
        MLMultiArray *output_result = model_output.result;
        // NSLog(@"%@", output_move);
        // NSLog(@"%@", output_result);

        fprintf(stderr, "Comparing output_policy to test case\n");
        check_result(output_policy_expected, (float*)output_move.dataPointer, batch_size * policy_size);
        fprintf(stderr, "Comparing output_value to test case\n");
        check_result(output_value_expected, (float*)output_result.dataPointer, batch_size * value_size);
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
