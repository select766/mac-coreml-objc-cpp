BATCH_SIZE=1
BACKEND=all
RUN_TIME_SEC=10

mainobjc: mainobjc.m DlShogiResnet15x224SwishBatch.m DlShogiResnet15x224SwishBatch.mlmodelc
	gcc -o mainobjc -O3 mainobjc.m DlShogiResnet15x224SwishBatch.m -framework Foundation -framework CoreML

mainobjcdynamic: mainobjcdynamic.m
	gcc -o mainobjcdynamic -O3 mainobjcdynamic.m -framework Foundation -framework CoreML

# gcc -o maincpp --std=c++11 maincpp.cpp nnwrapper.mm DlShogiResnet15x224SwishBatch.m -framework Foundation -framework CoreML
# error: invalid argument '--std=c++11' not allowed with 'Objective-C'
maincpp: maincpp.o nnwrapper.o DlShogiResnet15x224SwishBatch.o DlShogiResnet15x224SwishBatch.mlmodelc
	g++ -o maincpp maincpp.o nnwrapper.o DlShogiResnet15x224SwishBatch.o -framework Foundation -framework CoreML

maincpp.o: maincpp.cpp
	g++ -c --std=c++11 -O3 $<

nnwrapper.o: nnwrapper.mm
	g++ -c -O3 $<

DlShogiResnet15x224SwishBatch.o: DlShogiResnet15x224SwishBatch.m
	g++ -c -O3 $<

DlShogiResnet15x224SwishBatch.m: DlShogiResnet15x224SwishBatch.mlmodel
	/Applications/Xcode.app/Contents/Developer/usr/bin/coremlc generate DlShogiResnet15x224SwishBatch.mlmodel .

DlShogiResnet15x224SwishBatch.mlmodelc: DlShogiResnet15x224SwishBatch.mlmodel
	/Applications/Xcode.app/Contents/Developer/usr/bin/coremlc compile DlShogiResnet15x224SwishBatch.mlmodel .

runobjc: mainobjc
	./mainobjc ${BATCH_SIZE} ${BACKEND} ${RUN_TIME_SEC}

runcpp: maincpp
	./maincpp ${BATCH_SIZE} ${BACKEND} ${RUN_TIME_SEC}
