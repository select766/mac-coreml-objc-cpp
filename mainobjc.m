#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import "DlShogiResnet15x224SwishBatch.h"

int main(void) {
    MLModelConfiguration* config = [MLModelConfiguration new];
    config.computeUnits = MLComputeUnitsCPUAndGPU;

    NSError *error = nil;
    // カレントディレクトリの DlShogiResnet15x224SwishBatch.mlmodelc が読まれる (存在しないとmlmodel==null, error!=null)
    DlShogiResnet15x224SwishBatch* model = [[DlShogiResnet15x224SwishBatch alloc] initWithConfiguration:config error:&error];
    NSLog(@"%@", model);
    NSLog(@"%@", error);

    if (!model) {
        NSLog(@"Failed to load model, %@", error);
        exit(1);
    }

    return 0;
}
