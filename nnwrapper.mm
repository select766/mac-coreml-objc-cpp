// Objective-C, C++混在コード
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <stdlib.h>
#import "nnwrapper.hpp"
#import "DlShogiResnet15x224SwishBatch.h"

NNWrapper::NNWrapper(const char* computeUnits) {
    MLModelConfiguration* config = [MLModelConfiguration new];
    // 使用デバイス
    // MLComputeUnitsCPUOnly = 0,
    // MLComputeUnitsCPUAndGPU = 1,
    // MLComputeUnitsAll = 2
    MLComputeUnits cu;
    if (strcmp(computeUnits, "cpuonly") == 0) {
        cu = MLComputeUnitsCPUOnly;
    } else if (strcmp(computeUnits, "cpuandgpu") == 0) {
        cu = MLComputeUnitsCPUAndGPU;
    } else if (strcmp(computeUnits, "all") == 0) {
        cu = MLComputeUnitsAll;
    } else {
        fprintf(stderr, "Unknown computeUnits %s\n", computeUnits);
        exit(1);
    }
    config.computeUnits = cu;
    NSError *error = nil;
    DlShogiResnet15x224SwishBatch* model = [[DlShogiResnet15x224SwishBatch alloc] initWithConfiguration:config error:&error];
    NSLog(@"%@", model);
    NSLog(@"%@", error);

    if (!model) {
        NSLog(@"Failed to load model, %@", error);
        exit(1);
    }
    // 所有権をARCからプログラマに移す
    this->model = (void*)CFBridgingRetain(model);
}

bool NNWrapper::run(int batch_size, float* input, float* output_policy, float* output_value) {
    // 所有権を移さない(プログラマのまま)
    DlShogiResnet15x224SwishBatch* model = (__bridge DlShogiResnet15x224SwishBatch*)(this->model);

    MLMultiArray *model_input = [[MLMultiArray alloc] initWithDataPointer:input shape:@[[NSNumber numberWithInt:batch_size], @119, @9, @9] dataType:MLMultiArrayDataTypeFloat32 strides:@[@(119*9*9), @(9*9), @9, @1] deallocator:NULL error:NULL];

    NSError *error = nil;
    @autoreleasepool { // Core ML内部で確保されたメモリを解放するのに必要
        DlShogiResnet15x224SwishBatchOutput *model_output = [model predictionFromInput:model_input error:&error];
        if (error) {
            NSLog(@"%@", error);
            return false;
        }

        // 出力は動的確保された領域に書き出されるため、これを自前のバッファにコピー
        memcpy(output_policy, model_output.output_policy.dataPointer, batch_size * NNWrapper::policy_size * sizeof(float));
        memcpy(output_value, model_output.output_value.dataPointer, batch_size * NNWrapper::value_size * sizeof(float));
    }

    return true;
}

NNWrapper::~NNWrapper() {
    // 所有権をARCに返す
    DlShogiResnet15x224SwishBatch* model = CFBridgingRelease(this->model);
    // スコープを外れるので解放される
}
