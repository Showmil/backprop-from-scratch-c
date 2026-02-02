#include <stdio.h>  // 입출력 헤더 파일
#include <stdlib.h> // 유틸리티 함수 헤더 파일
#include <math.h>   // 수학 관련 함수 헤더 파일
#include <time.h>   // 시간 관련 함수

#define ANSI_RESET "\033[0m"
#define BATCH_SIZE 64      // batch 크기
#define EPOCH_SIZE 100     // epoch 크기
#define LEARNING_RATE 0.01 // 학습률 크기

/*
Deep Learning Information

input layer : 784 pixels
hidden layer1 : 512 neurons
    - active function : sigmoid
hidden layer2 : 256 neurons
    - active function : sigmoid
output layer : 10 newrons
    - active function : softmax
    - cost function : cross entropy

weight layer 1 : 512 * 784 weights
weight layer 2 : 256 * 512 weights
weight layer 3 : 10 * 256 weights

optimizer algorithm : Adam Optimizer
*/

// 2차원 동적 할당 배열 메모리 초기화 함수
void memoryFree(double **x, int n)
{
    for (int i = 0; i < BATCH_SIZE; i++)
    {
        free(x[i]);
    }
    free(x);
}

// 전체 이미지 데이터 저장함수
unsigned char **allImgData(int size, int count, FILE *f)
{
    // size: 이미지 하나당 픽셀 수 (784), count: 전체 이미지 개수 (60000)
    unsigned char **imgBuffer = (unsigned char **)malloc(sizeof(unsigned char *) * count);
    for (int i = 0; i < count; i++)
    {
        imgBuffer[i] = (unsigned char *)malloc(sizeof(unsigned char) * size);
    }

    // 이미지 데이터 픽셀 정보들을 배열에 저장
    for (int i = 0; i < count; i++)
    {
        fread(imgBuffer[i], sizeof(unsigned char), size, f);
    }

    // **imgBuffer가 아니라 포인터 변수 imgBuffer를 반환해야 함
    return imgBuffer;
}

// 이미지 데이터 전처리 함수
double **preprocessImgData(int size, int offset, unsigned char **imgData)
// size: input 크기(784), offset: 현재 배치의 시작 인덱스
{
    double **inputLayer = (double **)malloc(sizeof(double *) * BATCH_SIZE);
    for (int i = 0; i < BATCH_SIZE; i++)
    {
        inputLayer[i] = (double *)malloc(sizeof(double) * size);
    }

    for (int i = 0; i < BATCH_SIZE; i++)
    {
        for (int j = 0; j < size; j++)
        {
            inputLayer[i][j] = (double)imgData[offset + i][j] / 255.0;
        }
    }

    return inputLayer;
}

// 전체 라벨 데이터 저장 함수
unsigned char **allLabelData(int count, FILE *f)
{
    unsigned char **LabelBuffer = (unsigned char **)malloc(sizeof(unsigned char *) * count);
    for (int i = 0; i < count; i++)
    {
        LabelBuffer[i] = (unsigned char *)malloc(sizeof(unsigned char) * 1);
        fread(LabelBuffer[i], sizeof(unsigned char), 1, f);
    }

    return LabelBuffer;
}

// 라벨 데이터 전처리 함수
unsigned char **preprocessLabelData(int oneHotSize, int offset, unsigned char **allLabels)
// oneHotSize: 라벨 클래스 개수(10), offset: 현재 배치 시작 인덱스, allLabels: 전체 라벨 데이터
{
    unsigned char **labelData = (unsigned char **)malloc(sizeof(unsigned char *) * BATCH_SIZE);
    for (int i = 0; i < BATCH_SIZE; i++)
    {
        labelData[i] = (unsigned char *)malloc(sizeof(unsigned char) * oneHotSize);
    }

    for (int i = 0; i < BATCH_SIZE; i++)
    {
        unsigned char correctLabel = allLabels[offset + i][0];
        // One-Hot Encoding
        for (int j = 0; j < oneHotSize; j++)
        {
            if (j == correctLabel)
            {
                labelData[i][j] = 1;
            }
            else
            {
                labelData[i][j] = 0;
            }
        }
    }
    return labelData;
}

// MNIST 데이터셋의 헤더 비트가 뒤집어져 있어서 이를 뒤집어 주는 함수
int reverseInt(int i)
{
    unsigned int c1, c2, c3, c4 = 0; // 4바이트 정수를 1바이트 씩 쪼개기

    c1 = i & 255;         // 입력받은 정수 i의 가장 뒤쪽 8비트를 추출
    c2 = (i >> 8) & 255;  // 비트를 오른쪽으로 8칸 밀어서 1바이트를 추출
    c3 = (i >> 16) & 255; // 비트를 오른쪽으로 16칸 밀어서 1바이트를 추출
    c4 = (i >> 24) & 255; // 비트를 오른쪽으로 24칸 밀어서 가장 앞쪽 1바이트를 추출

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4; // 추출한 바이트들을 역순으로 재조립해 return!
}

// 가중치 레이어 생성 및 초기화 함수
double **createWeightLayer(int input, int output)
{
    // 가중치 레이어 동적 할당
    double **layer = (double **)malloc(sizeof(double *) * output);
    for (int i = 0; i < output; i++)
    {
        layer[i] = (double *)malloc(sizeof(double) * input);
    }

    // Xavier's uniform distribution 초기화 범위 계산
    double limit = sqrt(6.0 / ((double)input + (double)output));

    // Xavier 초기화
    for (int i = 0; i < output; i++)
    {
        for (int j = 0; j < input; j++)
        {
            // -limit ~ +limit 사이 랜덤 값
            layer[i][j] = ((double)rand() / RAND_MAX) * (2.0 * limit) - limit;
        }
    }

    return layer;
}

// 편향 행렬 생성 및 초기화 함수
double **createBiasLayer(int input, int output)
{
    // 편향 행렬 동적 할당
    double **layer = (double **)malloc(sizeof(double *) * BATCH_SIZE);
    for (int i = 0; i < BATCH_SIZE; i++)
    {
        layer[i] = (double *)malloc(sizeof(double) * output);
    }

    // Xavier's uniform distribution 초기화 범위 계산
    double limit = sqrt(6.0 / ((double)input + (double)output));

    // Xavier 초기화
    for (int k = 0; k < BATCH_SIZE; k++)
    {
        for (int i = 0; i < output; i++)
        {
            layer[k][i] = ((double)rand() / RAND_MAX) * (2.0 * limit) - limit;
        }
    }

    return layer;
}

// sigmoid 함수
double sigmoid(double n)
{
    return 1 / (1 + exp(-n));
}

// sigmoid 미분 함수
double sigmoidPrime(double n)
{
    return sigmoid(n) * (1 - sigmoid(n));
}

// softmax 함수
double **softmax(double **z, int size)
{
    double **a = (double **)malloc(sizeof(double *) * BATCH_SIZE);
    for (int i = 0; i < BATCH_SIZE; i++)
    {
        a[i] = (double *)malloc(sizeof(double) * size);
    }

    for (int k = 0; k < BATCH_SIZE; k++)
    {
        // 현재 배치에서 가장 큰 값 찾기
        double max = z[k][0];
        for (int i = 1; i < size; i++)
        {
            if (z[k][i] > max)
                max = z[k][i];
        }

        // max를 빼서 exp 연산하여 오버플로우를 방지
        double sum = 0.0;
        for (int i = 0; i < size; i++)
        {
            // z값에서 max를 빼도 확률 비율은 동일하게 유지됨
            a[k][i] = exp(z[k][i] - max);
            sum += a[k][i];
        }

        // 3. 나누기
        for (int i = 0; i < size; i++)
        {
            a[k][i] /= sum;
        }
    }
    return a;
}

// 활성화 함수
double **activate(double **z, int size, double (*func)(double)) // z: 활성화 함수에 넣을 대상, size: z의 크기, func: 활성화 함수
{
    double **a = (double **)malloc(sizeof(double *) * BATCH_SIZE);
    for (int i = 0; i < BATCH_SIZE; i++)
    {
        a[i] = (double *)malloc(sizeof(double) * size);
    }

    for (int k = 0; k < BATCH_SIZE; k++)
    {
        for (int i = 0; i < size; i++)
        {
            a[k][i] = func(z[k][i]);
        }
    }
    return a;
}

// 순전파 연산 함수
double **linear(double **w, double **x, double **b, int input, int output) // w: weight layer, x: input layer, b: bias layer, input: x의 node 개수, output: 출력되는 node 개수
{
    double **z = (double **)malloc(sizeof(double *) * BATCH_SIZE); // z 연산 결과 저장 배열
    for (int i = 0; i < BATCH_SIZE; i++)
    {
        z[i] = (double *)malloc(sizeof(double) * output);
    }

    for (int k = 0; k < BATCH_SIZE; k++)
    {
        for (int i = 0; i < output; i++)
        {
            z[k][i] = b[k][i]; // 편향 더해놓기
            for (int j = 0; j < input; j++)
            {
                z[k][i] += w[i][j] * x[k][j]; // 역전파 계산
            }
        }
    }
    return z;
}

// 출력층 노드 delta 함수
double **createOutputDelta(unsigned char **y, double **a, int size) // y: 정답값, a: 출력층 예측값
{
    double **d = (double **)malloc(sizeof(double *) * BATCH_SIZE);
    for (int i = 0; i < BATCH_SIZE; i++)
    {
        d[i] = (double *)malloc(sizeof(double) * size);
    }

    for (int k = 0; k < BATCH_SIZE; k++)
    {
        for (int i = 0; i < size; i++)
        {
            d[k][i] = a[k][i] - (double)y[k][i];
        }
    }
    return d;
}

// 은닉층 노드 delta 함수
double **createHiddenDelta(double **z, double **next, double **w, int currSize, int nextSize)
// z: 현재 레이어의 z값 배열, next: 다음 레이어의 델타값 배열, w: 두 레이어를 잇는 가중치
{
    double **d = (double **)malloc(sizeof(double *) * BATCH_SIZE);
    for (int i = 0; i < BATCH_SIZE; i++)
    {
        d[i] = (double *)malloc(sizeof(double) * currSize);
    }

    for (int k = 0; k < BATCH_SIZE; k++)
    {
        for (int i = 0; i < currSize; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < nextSize; j++)
            {
                sum += w[j][i] * next[k][j];
            }
            d[k][i] = sum * sigmoid(z[k][i]) * (1 - sigmoid(z[k][i]));
        }
    }
    return d;
}

// 경사 하강법 함수
double gradientDescent(double dL_dw, double w, double n)
// dL_dw : 기울기, w: 기존 w값, n: 학습률
{
    return w - n * dL_dw;
}

// 역전파 계산 함수
void backpropagation(double **x, double **delta, double **w, int inputSize, int deltaSize, double **b, double learningRate)
{
    for (int k = 0; k < BATCH_SIZE; k++)
    {
        for (int i = 0; i < inputSize; i++)
        {
            for (int j = 0; j < deltaSize; j++)
            {
                // 기울기를 BATCH_SIZE 로 나누어 평균을 구함
                double dL_dw = (delta[k][j] * x[k][i]) / (double)BATCH_SIZE;
                w[j][i] = gradientDescent(dL_dw, w[j][i], learningRate);
            }
        }
    }
    for (int k = 0; k < BATCH_SIZE; k++)
    {
        for (int i = 0; i < deltaSize; i++)
        {
            // 편향도 마찬가지로 나누기
            b[k][i] -= (learningRate * delta[k][i]) / (double)BATCH_SIZE;
        }
    }
}

int main()
{
    srand(time(NULL)); // 난수 초기화
    ////////////////////////// 이미지, 라벨 데이터 입력 ////////////////////////////
    FILE *img = fopen("train-images.idx3-ubyte", "rb");   // MNIST 이미지 데이터 입력
    FILE *label = fopen("train-labels.idx1-ubyte", "rb"); // MNIST 라벨 데이터 입력

    // 변수값 선언 및 초기화
    int dataMagicNumber, labelMagicNumber, imgCount, labelCount, imgWidth, imgHeight = 0;

    fread(&dataMagicNumber, sizeof(dataMagicNumber), 1, img);     // 이미지 데이터의 매직넘버 부분 추출
    fread(&labelMagicNumber, sizeof(labelMagicNumber), 1, label); // 라벨 데이터의 매직넘버 부분 추출
    dataMagicNumber = reverseInt(dataMagicNumber);                // 매직넘버 할당
    labelMagicNumber = reverseInt(labelMagicNumber);              // 매직넘버 할당

    fread(&imgCount, sizeof(imgCount), 1, img);       // 이미지 데이터 개수 추출
    fread(&labelCount, sizeof(labelCount), 1, label); // 라벨 데이터의 개수 추출
    imgCount = reverseInt(imgCount);                  // 이미지 데이터 개수 할당
    labelCount = reverseInt(labelCount);              // 라벨 데이터 개수 할당

    fread(&imgWidth, sizeof(imgWidth), 1, img); // MNIST 이미지 헤더의 이미지 가로 크기 부분 추출
    imgWidth = reverseInt(imgWidth);            // 변수에 가로 크기 할당

    fread(&imgHeight, sizeof(imgHeight), 1, img); // MNIST 이미지 헤더의 이미지 세로 크기 부분 추출
    imgHeight = reverseInt(imgHeight);            // 변수에 세로 크기 할당

    int imgSize = imgWidth * imgHeight; // 이미지 크기 저장
    int labelSize = 10;                 // 라벨 데이터 개수(0~9로 10개)

    unsigned char **allImage = allImgData(imgSize, imgCount, img); // 전체 이미지 데이터 저장
    unsigned char **allLabels = allLabelData(labelCount, label);   // 라벨 데이터 저장
    ////////////////////////////////////////////////////////////////////////////

    //////////////////////// 가중치, 편향 행렬 생성 및 초기화 //////////////////////////
    int hiddenLayer1Count = 512; // 은닉층1 뉴런 개수
    int hiddenLayer2Count = 256; // 은닉층2 뉴런 개수
    int outputLayerCount = 10;   // 출력층 뉴런 개수

    double **weightLayer1 = createWeightLayer(imgSize, hiddenLayer1Count);           // weight layer 1(512 * 784) 생성
    double **weightLayer2 = createWeightLayer(hiddenLayer1Count, hiddenLayer2Count); // weight layer 2(256 * 512) 생성
    double **weightLayer3 = createWeightLayer(hiddenLayer2Count, outputLayerCount);  // weight layer 3(10 * 256) 생성

    double **biasLayer1 = createBiasLayer(imgSize, hiddenLayer1Count);           // bias layer 1(512 * 1) 생성
    double **biasLayer2 = createBiasLayer(hiddenLayer1Count, hiddenLayer2Count); // bias layer 2(256 * 1) 생성
    double **biasLayer3 = createBiasLayer(hiddenLayer2Count, outputLayerCount);  // bias layer 3(10 * 1) 생성
    ////////////////////////////////////////////////////////////////////////////

    for (int e = 0; e < EPOCH_SIZE; e++)
    { // EPOCH 만큼 반복
        printf("------EPOCH %d------\n", e + 1);
        for (int i = 0; i < imgCount / BATCH_SIZE; i++) //
        {
            printf("--%d ~ %d 데이터로 학습--\n", i * BATCH_SIZE, BATCH_SIZE * (i + 1));
            //////////////////////// 이미지, 라벨 데이터 전처리 //////////////////////////
            double offset = i * BATCH_SIZE;
            double **inputLayer = preprocessImgData(imgSize, offset, allImage);            // 이미지 데이터 전처리
            unsigned char **labelData = preprocessLabelData(labelSize, offset, allLabels); // 라벨 데이터 전처리
            ////////////////////////////////////////////////////////////////////////////

            ///////////////////// Forward Propagation(순전파) 연산 //////////////////////
            double **z1 = linear(weightLayer1, inputLayer, biasLayer1, imgSize, hiddenLayer1Count);
            double **a1 = activate(z1, hiddenLayer1Count, sigmoid);
            double **z2 = linear(weightLayer2, a1, biasLayer2, hiddenLayer1Count, hiddenLayer2Count);
            double **a2 = activate(z2, hiddenLayer2Count, sigmoid);
            double **z3 = linear(weightLayer3, a2, biasLayer3, hiddenLayer2Count, outputLayerCount);
            double **a3 = softmax(z3, outputLayerCount);
            ////////////////////////////////////////////////////////////////////////////

            ///////////////////// Back Propagation(역전파) 연산 /////////////////////////
            double **outputDelta = createOutputDelta(labelData, a3, outputLayerCount);
            double **hiddenLayer2Delta = createHiddenDelta(z2, outputDelta, weightLayer3, hiddenLayer2Count, outputLayerCount);
            double **hiddenLayer1Delta = createHiddenDelta(z1, hiddenLayer2Delta, weightLayer2, hiddenLayer1Count, hiddenLayer2Count);

            backpropagation(a2, outputDelta, weightLayer3, hiddenLayer2Count, outputLayerCount, biasLayer3, LEARNING_RATE);
            backpropagation(a1, hiddenLayer2Delta, weightLayer2, hiddenLayer1Count, hiddenLayer2Count, biasLayer2, LEARNING_RATE);
            backpropagation(inputLayer, hiddenLayer1Delta, weightLayer1, imgSize, hiddenLayer1Count, biasLayer1, LEARNING_RATE);
            ////////////////////////////////////////////////////////////////////////////

            ////////////////////////////// 메모리 초기화 ////////////////////////////////
            memoryFree(inputLayer, BATCH_SIZE);
            memoryFree(z1, BATCH_SIZE);
            memoryFree(a1, BATCH_SIZE);
            memoryFree(z2, BATCH_SIZE);
            memoryFree(a2, BATCH_SIZE);
            memoryFree(z3, BATCH_SIZE);
            memoryFree(a3, BATCH_SIZE);
            memoryFree(outputDelta, BATCH_SIZE);
            memoryFree(hiddenLayer2Delta, BATCH_SIZE);
            memoryFree(hiddenLayer1Delta, BATCH_SIZE);
            for (int i = 0; i < BATCH_SIZE; i++)
                free(labelData[i]);
            free(labelData);
            ////////////////////////////////////////////////////////////////////////////
        }
    }

    ///////////////////////////////// TEST ////////////////////////////////////
    FILE *testImg = fopen("t10k-images.idx3-ubyte", "rb");   // MNIST 테스트 이미지 데이터 입력
    FILE *testLabel = fopen("t10k-labels.idx1-ubyte", "rb"); // MNIST 테스트 라벨 데이터 입력

    // 테스트 데이터 변수값 선언 및 초기화
    int testDataMagicNumber, testLabelMagicNumber, testImgCount, testLabelCount, testImgWidth, testImgHeight = 0;

    fread(&testDataMagicNumber, sizeof(testDataMagicNumber), 1, testImg); // 매직넘버 추출
    fread(&testLabelMagicNumber, sizeof(testLabelMagicNumber), 1, testLabel);
    testDataMagicNumber = reverseInt(testDataMagicNumber);
    testLabelMagicNumber = reverseInt(testLabelMagicNumber);

    fread(&testImgCount, sizeof(testImgCount), 1, testImg);       // 데이터 개수 추출
    fread(&testLabelCount, sizeof(testLabelCount), 1, testLabel); // 라벨 데이터의 개수 추출
    testImgCount = reverseInt(testImgCount);
    testLabelCount = reverseInt(testLabelCount);

    fread(&testImgWidth, sizeof(testImgWidth), 1, testImg); // 가로 크기 추출
    testImgWidth = reverseInt(testImgWidth);

    fread(&testImgHeight, sizeof(testImgHeight), 1, testImg); // 세로 크기 추출
    testImgHeight = reverseInt(testImgHeight);

    int testImgSize = testImgWidth * testImgHeight; // 이미지 크기 저장

    unsigned char **testAllImage = allImgData(testImgSize, testImgCount, testImg);    // 전체 이미지 데이터 저장
    unsigned char **testAllLabels = allLabelData(testLabelCount, testLabel);          // 라벨 데이터 저장
    double **testInputLayer = preprocessImgData(imgSize, 0, testAllImage);            // 이미지 데이터 전처리
    unsigned char **testLabelData = preprocessLabelData(labelSize, 0, testAllLabels); // 라벨 데이터 전처리

    double **test_z1 = linear(weightLayer1, testInputLayer, biasLayer1, imgSize, hiddenLayer1Count);
    double **test_a1 = activate(test_z1, hiddenLayer1Count, sigmoid);
    double **test_z2 = linear(weightLayer2, test_a1, biasLayer2, hiddenLayer1Count, hiddenLayer2Count);
    double **test_a2 = activate(test_z2, hiddenLayer2Count, sigmoid);
    double **test_z3 = linear(weightLayer3, test_a2, biasLayer3, hiddenLayer2Count, outputLayerCount);
    double **test_a3 = softmax(test_z3, outputLayerCount);
    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////// CLI 출력 부분 ///////////////////////////////
    // 이미지 데이터 출력
    for (int y = 0; y < testImgHeight; y++)
    {
        for (int x = 0; x < testImgWidth; x++)
        {
            unsigned char pixel = (unsigned char)(testInputLayer[0][y * testImgWidth + x] * 255.0); // 현재 픽셀의 위치 정보
            if (x < 2 || x >= testImgWidth - 2)
            {
                printf(ANSI_RESET "  "); // 이미지 깨짐 문제 방지 위해 양 옆 2개의 픽셀 초기화
            }
            else
            {
                printf("\033[48;2;%d;%d;%dm  ", pixel, pixel, pixel); // 이미지 출력
            }
        }
        printf(ANSI_RESET "\n"); // 줄바꾸면서 초기화
    }

    // 정답 라벨 출력
    int ans = 0;
    for (int i = 0; i < labelSize; i++)
    {
        if (testLabelData[0][i] == 1)
        {
            ans = i;
            break;
        }
    }
    printf("answer : %d\n", ans);

    // 0~9 확률 출력
    for (int i = 0; i < labelSize; i++)
    {
        printf("%d 일 확률 : %f\n", i, test_a3[0][i]);
    }
    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////// 메모리 해제 //////////////////////////////////

    for (int i = 0; i < hiddenLayer1Count; i++)
        free(weightLayer1[i]);
    free(weightLayer1);

    for (int i = 0; i < hiddenLayer2Count; i++)
        free(weightLayer2[i]);
    free(weightLayer2);

    for (int i = 0; i < outputLayerCount; i++)
        free(weightLayer3[i]);
    free(weightLayer3);

    for (int i = 0; i < BATCH_SIZE; i++)
    {
        free(biasLayer1[i]);
    }
    free(biasLayer1);

    for (int i = 0; i < BATCH_SIZE; i++)
    {
        free(biasLayer2[i]);
    }
    free(biasLayer2);

    for (int i = 0; i < BATCH_SIZE; i++)
    {
        free(biasLayer3[i]);
    }
    free(biasLayer3);

    fclose(img);
    fclose(label);
    ////////////////////////////////////////////////////////////////////////////

    system("pause");
}