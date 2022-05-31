# mac-coreml-objc-cpp
MacのCore MLをObjective-C及びC++から使うサンプル

dlshogiの評価関数モデルを動作させる。

macOS 12.4で動作確認。Intel Mac / M1 Mac両対応。

# Objective-Cだけで完結する版

ソースコード: `mainobjc.m`

## ビルド
```
make mainobjc
```

ビルドの過程で、モデルのコンパイル(`DlShogiResnet15x224SwishBatch.mlmodelc`ディレクトリ)及びモデルをロードするためのコード(`DlShogiResnet15x224SwishBatch.[hm]`)が自動生成される。

## 実行

バッチサイズ1、全てのバックエンドを利用、10秒間計測する場合

```
./mainobjc 1 all 10
```

バックエンドは、 `all`: 全てのデバイス(Neural Engine/GPU/CPU)を利用、`cpuandgpu`: CPUまたはGPUを利用、`cpuonly`: CPUのみ利用のいずれか。

# C++から呼び出す版

ソースコード: `maincpp.cpp`, `nnwrapper.mm`, `nnwrapper.hpp`

`maincpp.cpp`は純粋なC++コードで、Objective-Cのことを気にせずCore MLを呼び出すことができる。

## ビルド
```
make maincpp
```

## 実行

バッチサイズ1、全てのバックエンドを利用、10秒間計測する場合

```
./maincpp 1 all 10
```

C++版の方が、Objective-C版より若干パフォーマンスが低下する。C++版では出力データを独自のバッファにコピーするが、Objective-C版ではしない（計算結果を無視）ためであると考えられる。

# モデル・テストケースの作成方法

「強い将棋ソフトの創りかた」サンプルコードに従い、15ブロック224チャンネル(`resnet15x224_swish`)のモデルを学習後、以下のリポジトリのjupyter notebookを用いてmlmodelモデルファイルおよびテストケースを生成している。

https://github.com/select766/dlshogi-model-on-coreml/tree/master/colab

可変バッチサイズに対応したモデルの作り方は https://select766.hatenablog.com/entry/2022/01/29/190100 参照。

## 結果のずれについて

PyTorchで作成したテストケースと結果を比較する機能が実装されている。

`cpuonly`の場合の例

```
Comparing output_policy to test case
Max difference: 0.000025
Comparing output_value to test case
Max difference: 0.000001
```

`cpuonly`以外の場合の例

```
Comparing output_policy to test case
Error at index 49: -0.918101 != -1.128906
Comparing output_value to test case
Max difference: 0.007170
```

policyについて、index0〜48は許容範囲だが、49は大きな誤差が出ている。内部の計算がfloat16であるため、桁落ちなどの要因で大きな誤差が出る場合があるようである。
