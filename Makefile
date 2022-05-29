BATCH_SIZE=1
BACKEND=all
RUN_TIME_SEC=10

mainobjc: mainobjc.m DlShogiResnet15x224SwishBatch.m DlShogiResnet15x224SwishBatch.mlmodelc
	gcc -o mainobjc mainobjc.m DlShogiResnet15x224SwishBatch.m -framework Foundation -framework CoreML

DlShogiResnet15x224SwishBatch.m: DlShogiResnet15x224SwishBatch.mlmodel
	/Applications/Xcode.app/Contents/Developer/usr/bin/coremlc generate DlShogiResnet15x224SwishBatch.mlmodel .

DlShogiResnet15x224SwishBatch.mlmodelc: DlShogiResnet15x224SwishBatch.mlmodel
	/Applications/Xcode.app/Contents/Developer/usr/bin/coremlc compile DlShogiResnet15x224SwishBatch.mlmodel .

run: mainobjc
	./mainobjc ${BATCH_SIZE} ${BACKEND} ${RUN_TIME_SEC}
