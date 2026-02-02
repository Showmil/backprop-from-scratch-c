#include <stdio.h>  // 입출력 헤더 파일
#include <stdlib.h> // 유틸리티 함수 헤더 파일
#include <math.h>   // 수학 관련 함수 헤더 파일
#include <time.h>   // 시간 관련 함수

#define ANSI_RESET "\033[0m"

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
double *createBiasLayer(int input, int output)
{
    // 편향 행렬 동적 할당
    double *layer = (double *)malloc(sizeof(double) * output);

    // Xavier's uniform distribution 초기화 범위 계산
    double limit = sqrt(6.0 / ((double)input + (double)output));

    // Xavier 초기화
    for (int i = 0; i < output; i++)
    {
        layer[i] = ((double)rand() / RAND_MAX) * (2.0 * limit) - limit;
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
double *softmax(double *z, int size)
{
    double sum = 0.0;
    double *a = (double *)malloc(sizeof(double) * size);

    for (int i = 0; i < size; i++)
    {
        sum += exp(z[i]);
    }

    for (int i = 0; i < size; i++)
    {
        a[i] = exp(z[i]) / sum;
    }

    return a;
}

// 활성화 함수
double *activate(double *z, int size, double (*func)(double)) // z: 활성화 함수에 넣을 대상, size: z의 크기, func: 활성화 함수
{
    double *a = (double *)malloc(sizeof(double) * size);
    for (int i = 0; i < size; i++)
    {
        a[i] = func(z[i]);
    }
    return a;
}

// 순전파 연산 함수
double *linear(double **w, double *x, double *b, int input, int output) // w: weight layer, x: input layer, b: bias layer, input: x의 node 개수, output: 출력되는 node 개수
{
    double *z = (double *)malloc(sizeof(double) * output); // z 연산 결과 저장 배열
    for (int i = 0; i < output; i++)
    {
        z[i] = 0; // 0으로 배열 초기화
    }

    for (int i = 0; i < output; i++)
    {
        for (int j = 0; j < input; j++)
        {
            z[i] += w[i][j] * x[j] + b[i]; // 역전파 계산
        }
    }

    return z;
}

// 출력층 노드 delta 함수
double *createOutputDelta(unsigned char *y, double *a, int size) // y: 정답값, a: 출력층 예측값
{
    double *d = (double *)malloc(sizeof(double) * size);
    for (int i = 0; i < size; i++)
    {
        printf("y[%d] : %d\n", i, y[i]);
        printf("a[%d] : %.4f\n", i, a[i]);
        d[i] = a[i] - (double)y[i];
    }
    return d;
}

// 은닉층 노드 delta 함수
double *createHiddenDelta(double *z, double *next, double **w, int currSize, int nextSize)
// z: 현재 레이어의 z값 배열, next: 다음 레이어의 델타값 배열, w: 두 레이어를 잇는 가중치
{
    double *d = (double *)malloc(sizeof(double) * currSize);
    for (int i = 0; i < currSize; i++)
    {
        double sum = 0.0;
        for (int j = 0; j < nextSize; j++)
        {
            sum += w[j][i] * next[j];
        }
        d[i] = sum * sigmoid(z[i]) * (1 - sigmoid(z[i]));
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
void backpropagation(double *x, double *delta, double **w, int inputSize, int deltaSize)
// x: 입력값, delta: 델타값, w: 가중치 배열
{
    for (int i = 0; i < inputSize; i++)
    {
        for (int j = 0; j < deltaSize; j++)
        {
            double dL_dw = delta[j] * x[i];
            w[j][i] = gradientDescent(dL_dw, w[j][i], 0.01);
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
    int batchSize = 30;                 // 사용할 이미지 개수
    int labelSize = 10;                 // 라벨 데이터 개수(0~9로 10개)

    printf("Train 이미지 개수 출력 : %d\n", imgCount); // Train 이미지 개수 출력
    ////////////////////////////////////////////////////////////////////////////

    //////////////////////// 이미지, 라벨 데이터 전처리 //////////////////////////
    // 이미지 데이터 저장 배열 생성
    unsigned char **imgBuffer = (unsigned char **)malloc(sizeof(unsigned char *) * batchSize);
    for (int i = 0; i < batchSize; i++)
    { // 784개의 개별 이미지 픽셀 정보를 저장할 배열 동적 할당
        imgBuffer[i] = (unsigned char *)malloc(sizeof(unsigned char) * imgSize);
    }

    // 이미지 데이터 픽셀 정보들을 배열에 저장
    for (int i = 0; i < batchSize; i++)
    {
        fread(imgBuffer[i], sizeof(unsigned char), imgSize, img);
    }

    // 이미지 데이터 출력
    for (int y = 0; y < imgHeight; y++)
    {
        for (int x = 0; x < imgWidth; x++)
        {
            unsigned char pixel = imgBuffer[0][y * imgWidth + x]; // 현재 픽셀의 위치 정보
            if (x < 2 || x >= imgWidth - 2)
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

    // 라벨 데이터 저장 배열 생성
    unsigned char **labelData = (unsigned char **)malloc(sizeof(unsigned char *) * batchSize);
    for (int i = 0; i < batchSize; i++)
    { // 10개의 라벨 데이터 정보를 저장할 배열 동적 할당
        labelData[i] = (unsigned char *)malloc(sizeof(unsigned char) * labelSize);
    }

    unsigned char tempLabel = 0; // 파일에서 읽어올 1바이트 분량의 임시 변수
    // 파일에서 실제 정답 라벨 1개를 읽기
    fread(&tempLabel, sizeof(unsigned char), 1, label);
    // One-Hot Encoding 변환
    for (int j = 0; j < 10; j++)
    {
        if (j == tempLabel)
        {
            labelData[0][j] = 1; // 정답 인덱스에는 1
        }
        else
        {
            labelData[0][j] = 0; // 나머지 인덱스에는 0
        }
    }

    // 라벨 데이터 출력
    for (int k = 0; k < labelSize; k++)
    {
        printf("%d ", labelData[0][k]);
    }
    printf("\n");
    ////////////////////////////////////////////////////////////////////////////

    //////////////////////// 가중치, 편향 행렬 생성 및 초기화 //////////////////////////
    // weight layer 1 : 512 * 784 weights
    int hiddenLayer1Count = 512; // 은닉층1 뉴런 개수
    int hiddenLayer2Count = 256; // 은닉층2 뉴런 개수
    int outputLayerCount = 10;   // 출력층 뉴런 개수

    double **inputLayer = (double **)malloc(sizeof(double *) * batchSize);
    for (int i = 0; i < batchSize; i++)
    {
        inputLayer[i] = (double *)malloc(sizeof(double) * imgSize);
    }

    for (int i = 0; i < batchSize; i++) // Input Layer가 unsigned char형이라 double로 전처리
    {
        for (int j = 0; j < imgSize; j++)
        {
            inputLayer[i][j] = (double)imgBuffer[i][j] / 255.0; // 기울기 소실 방지 위해 255로 나누어 0~1 값으로 정규화
        }
    }

    double **weightLayer1 = createWeightLayer(imgSize, hiddenLayer1Count);           // weight layer 1(512 * 784) 생성
    double **weightLayer2 = createWeightLayer(hiddenLayer1Count, hiddenLayer2Count); // weight layer 2(256 * 512) 생성
    double **weightLayer3 = createWeightLayer(hiddenLayer2Count, outputLayerCount);  // weight layer 3(10 * 256) 생성

    double *biasLayer1 = createBiasLayer(imgSize, hiddenLayer1Count);           // bias layer 1(512 * 1) 생성
    double *biasLayer2 = createBiasLayer(hiddenLayer1Count, hiddenLayer2Count); // bias layer 2(256 * 1) 생성
    double *biasLayer3 = createBiasLayer(hiddenLayer2Count, outputLayerCount);  // bias layer 3(10 * 1) 생성

    // weight layer 1(512 * 784) 출력
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            printf("weight(input node[%d] -> hidden layer 1[%d]) : %.4f\n", j, i, weightLayer1[i][j]);
        }
    }

    // weight layer 2(256 * 512) 출력
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            printf("weight(hidden layer 1[%d] -> hidden layer 2[%d]) : %.4f\n", j, i, weightLayer2[i][j]);
        }
    }

    // weight layer 3(10 * 256) 출력
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            printf("weight(hidden layer 2[%d] -> output layer[%d]) : %.4f\n", j, i, weightLayer3[i][j]);
        }
    }

    /*
    // bias layer 1(512 * 1) 출력
    for (int i = 0; i < 5; i++)
    {
        printf("bias - hidden layer 1[%d] : %.4f\n", i, biasLayer1[i]);
    }

    // bias layer 2(256 * 1) 출력
    for (int i = 0; i < 5; i++)
    {
        printf("bias - hidden layer 2[%d] : %.4f\n", i, biasLayer2[i]);
    }

    // bias layer 3(10 * 1) 출력
    for (int i = 0; i < 5; i++)
    {
        printf("bias - output layer[%d] : %.4f\n", i, biasLayer3[i]);
    }
    */

    ////////////////////////////////////////////////////////////////////////////

    ///////////////////// Forward Propagation(순전파) 연산 //////////////////////
    // 1. z1 = weight layer 1 * imgBuffer[0] + bias 1 계산
    double *z1 = linear(weightLayer1, inputLayer[0], biasLayer1, imgSize, hiddenLayer1Count);
    // 2. a1 = sigmoid * z1 계산
    double *a1 = activate(z1, hiddenLayer1Count, sigmoid);
    // 3. z2 = weight layer 2 * a1 + bias 2 계산
    double *z2 = linear(weightLayer2, a1, biasLayer2, hiddenLayer1Count, hiddenLayer2Count);
    // 4. a2 = sigmoid * z2 계산
    double *a2 = activate(z2, hiddenLayer2Count, sigmoid);
    // 5. z3 = weight layer 3 * a2 + bias 3 계산
    double *z3 = linear(weightLayer3, a2, biasLayer3, hiddenLayer2Count, outputLayerCount);
    // 6. a3 = softmax * z3 계산
    double *a3 = softmax(z3, outputLayerCount);

    /*
    for (int i = 0; i < 5; i++)
    {
        printf("z1[%d] : %.4f\n", i, z1[i]);
    }
    for (int i = 0; i < 5; i++)
    {
        printf("a1[%d] : %.4f\n", i, a1[i]);
    }
    for (int i = 0; i < 5; i++)
    {
        printf("z2[%d] : %.4f\n", i, z2[i]);
    }
    for (int i = 0; i < 5; i++)
    {
        printf("a2[%d] : %.4f\n", i, a2[i]);
    }
    for (int i = 0; i < 5; i++)
    {
        printf("z3[%d] : %.4f\n", i, z3[i]);
    }
    for (int i = 0; i < 10; i++)
    {
        printf("a3[%d] : %.4f\n", i, a3[i]);
    }
    */

    ////////////////////////////////////////////////////////////////////////////

    ///////////////////// Back Propagation(역전파) 연산 ////////////////////////
    // printf("--------------------------delta------------------------------------\n");

    double *outputDelta = createOutputDelta(labelData[0], a3, outputLayerCount);
    double *hiddenLayer2Delta = createHiddenDelta(z2, outputDelta, weightLayer3, hiddenLayer2Count, outputLayerCount);
    double *hiddenLayer1Delta = createHiddenDelta(z1, hiddenLayer2Delta, weightLayer2, hiddenLayer1Count, hiddenLayer2Count);

    for (int i = 0; i < 10; i++)
    {
        printf("outputDelta[%d] : %.4f\n", i, outputDelta[i]);

    }
    for (int i = 0; i < 4; i++)
    {
        printf("hiddenLayer2Delta[%d] : %.4f\n", i, hiddenLayer2Delta[i]);
        
    }
    for (int i = 0; i < 4; i++)
    {
        printf("hiddenLayer1Delta[%d] : %.4f\n", i, hiddenLayer1Delta[i]);
        
    }

    backpropagation(a2, outputDelta, weightLayer3, hiddenLayer2Count, outputLayerCount);
    backpropagation(a1, hiddenLayer2Delta, weightLayer2, hiddenLayer1Count, hiddenLayer2Count);
    backpropagation(inputLayer[0], hiddenLayer1Delta, weightLayer1, imgSize, hiddenLayer1Count);

    // weight layer 1(512 * 784) 출력
    printf("--------------------------Backpropagation------------------------------------\n");
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            printf("weight(input node[%d] -> hidden layer 1[%d]) : %.4f\n", j, i, weightLayer1[i][j]);
        }
    }

    // weight layer 2(256 * 512) 출력
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            printf("weight(hidden layer 1[%d] -> hidden layer 2[%d]) : %.4f\n", j, i, weightLayer2[i][j]);
        }
    }

    // weight layer 3(10 * 256) 출력
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            printf("weight(hidden layer 2[%d] -> output layer[%d]) : %.4f\n", j, i, weightLayer3[i][j]);
        }
    }

    ////////////////////////////////////////////////////////////////////////////

    /////////////// 나머지 9999개의 이미지 행렬화 해서 빠르게 연산 ////////////////

    ////////////////////////////////////////////////////////////////////////////

    ////////////////////// 테스트 데이터셋 이용해서 테스트 ///////////////////////

    ////////////////////////////// 메모리 해제 //////////////////////////////////
    for (int i = 0; i < batchSize; i++)
        free(imgBuffer[i]);
    free(imgBuffer);

    for (int i = 0; i < batchSize; i++)
        free(labelData[i]);
    free(labelData);

    for (int i = 0; i < hiddenLayer1Count; i++)
        free(weightLayer1[i]);
    free(weightLayer1);

    for (int i = 0; i < hiddenLayer2Count; i++)
        free(weightLayer2[i]);
    free(weightLayer2);

    for (int i = 0; i < outputLayerCount; i++)
        free(weightLayer3[i]);
    free(weightLayer3);

    free(biasLayer1);
    free(biasLayer2);
    free(biasLayer3);

    free(z1);
    free(a1);
    free(z2);
    free(a2);
    free(z3);
    free(a3);

    fclose(img);
    fclose(label);
    ////////////////////////////////////////////////////////////////////////////

    system("pause");
}