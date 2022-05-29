objc: mainobjc.m DlShogiResnet15x224SwishBatch.m DlShogiResnet15x224SwishBatch.mlmodelc
	gcc -o mainobjc mainobjc.m -framework Foundation

DlShogiResnet15x224SwishBatch.m: DlShogiResnet15x224SwishBatch.mlmodel
	/Applications/Xcode.app/Contents/Developer/usr/bin/coremlc generate DlShogiResnet15x224SwishBatch.mlmodel .

DlShogiResnet15x224SwishBatch.mlmodelc: DlShogiResnet15x224SwishBatch.mlmodel
	/Applications/Xcode.app/Contents/Developer/usr/bin/coremlc compile DlShogiResnet15x224SwishBatch.mlmodel .
