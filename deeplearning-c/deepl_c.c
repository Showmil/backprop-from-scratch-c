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

int main()
{
    srand(time(NULL)); // 난수 초기화
    ////////////////////////// 이미지, 라벨 데이터 입력 ////////////////////////////
    FILE *img = fopen("t10k-images.idx3-ubyte", "rb");   // MNIST 이미지 데이터 입력
    FILE *label = fopen("t10k-labels.idx1-ubyte", "rb"); // MNIST 라벨 데이터 입력

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
    unsigned char **labelBuffer = (unsigned char **)malloc(sizeof(unsigned char *) * batchSize);
    for (int i = 0; i < batchSize; i++)
    { // 10개의 라벨 데이터 정보를 저장할 배열 동적 할당
        labelBuffer[i] = (unsigned char *)malloc(sizeof(unsigned char) * labelSize);
    }

    unsigned char tempLabel = 0; // 파일에서 읽어올 1바이트 분량의 임시 변수
    // 파일에서 실제 정답 라벨 1개를 읽기
    fread(&tempLabel, sizeof(unsigned char), 1, label);
    // One-Hot Encoding 변환
    for (int j = 0; j < 10; j++)
    {
        if (j == tempLabel)
        {
            labelBuffer[0][j] = 1; // 정답 인덱스에는 1
        }
        else
        {
            labelBuffer[0][j] = 0; // 나머지 인덱스에는 0
        }
    }

    // 라벨 데이터 출력
    for (int k = 0; k < labelSize; k++)
    {
        printf("%d ", labelBuffer[0][k]);
    }
    printf("\n");
    ////////////////////////////////////////////////////////////////////////////

    //////////////////////// 가중치, 편향 행렬 생성 및 초기화 //////////////////////////
    // weight layer 1 : 512 * 784 weights
    int hiddenLayer1Count = 512; // 은닉층1 뉴런 개수
    int hiddenLayer2Count = 256; // 은닉층2 뉴런 개수
    int outputLayerCount = 10;   // 출력층 뉴런 개수

    // 가중치 레이어1 512 * 784 배열 동적 할당
    double **weightLayer1 = (double **)malloc(sizeof(double *) * hiddenLayer1Count);
    for (int i = 0; i < hiddenLayer1Count; i++)
    {
        weightLayer1[i] = (double *)malloc(sizeof(double) * (imgWidth * imgHeight));
    }

    // 가중치 레이어1 Xavier's uniform distribution으로 초기화
    double n1_input = (double)(imgWidth * imgHeight);   // 784
    double n1_output = (double)hiddenLayer1Count;       // 512
    double limit1 = sqrt(6.0 / (n1_input + n1_output)); // Xavier 초기화의 범위

    for (int i = 0; i < hiddenLayer1Count; i++)
    {
        for (int j = 0; j < (imgWidth * imgHeight); j++)
        {
            weightLayer1[i][j] = ((double)rand() / RAND_MAX) * (2.0 * limit1) - limit1; // Xavier 범위 내의 랜덤 weight 할당
        }
    }

    // 가중치 레이어1 출력
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            printf("weight(input node[%d] -> hidden layer 1[%d]) : %.4f\n", j, i, weightLayer1[i][j]);
        }
    }

    // 가중치 레이어2 256 * 512 배열 동적 할당
    double **weightLayer2 = (double **)malloc(sizeof(double *) * hiddenLayer2Count);
    for (int i = 0; i < hiddenLayer2Count; i++)
    {
        weightLayer2[i] = (double *)malloc(sizeof(double) * hiddenLayer1Count);
    }

    // 가중치 레이어2 Xavier's uniform distribution으로 초기화
    double n2_input = (double)hiddenLayer1Count;        // 512
    double n2_output = (double)hiddenLayer2Count;       // 256
    double limit2 = sqrt(6.0 / (n2_input + n2_output)); // Xavier 초기화의 범위

    for (int i = 0; i < hiddenLayer2Count; i++)
    {
        for (int j = 0; j < hiddenLayer1Count; j++)
        {
            weightLayer2[i][j] = ((double)rand() / RAND_MAX) * (2.0 * limit2) - limit2; // Xavier 범위 내의 랜덤 weight 할당
        }
    }

    // 가중치 레이어2 출력
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            printf("weight(hidden layer 1[%d] -> hidden layer 2[%d]) : %.4f\n", j, i, weightLayer2[i][j]);
        }
    }

    // 가중치 레이어3 10 * 256 배열 동적 할당
    double **weightLayer3 = (double **)malloc(sizeof(double *) * outputLayerCount);
    for (int i = 0; i < outputLayerCount; i++)
    {
        weightLayer3[i] = (double *)malloc(sizeof(double) * hiddenLayer2Count);
    }

    // 가중치 레이어3 Xavier's uniform distribution으로 초기화
    double n3_input = (double)hiddenLayer2Count;        // 256
    double n3_output = (double)outputLayerCount;        // 10
    double limit3 = sqrt(6.0 / (n3_input + n3_output)); // Xavier 초기화의 범위

    for (int i = 0; i < outputLayerCount; i++)
    {
        for (int j = 0; j < hiddenLayer2Count; j++)
        {
            weightLayer3[i][j] = ((double)rand() / RAND_MAX) * (2.0 * limit3) - limit3; // Xavier 범위 내의 랜덤 weight 할당
        }
    }

    // 가중치 레이어3 출력
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            printf("weight(hidden layer 2[%d] -> output layer[%d]) : %.4f\n", j, i, weightLayer3[i][j]);
        }
    }

    // 편향 행렬1 512 * 1 동적 할당

    // 편향 행렬1 Xavier's uniform distribution으로 초기화

    // 편향 행렬1 출력

    // 편향 행렬2 256 * 1 동적 할당

    // 편향 행렬2 Xavier's uniform distribution으로 초기화

    // 편향 행렬2 출력

    // 편향 행렬3 10 * 1 동적 할당

    // 편향 행렬3 Xavier's uniform distribution으로 초기화

    // 편향 행렬3 출력

    ////////////////////////////////////////////////////////////////////////////

    ///////////////////// Forward Propagation(순전파) 연산 //////////////////////
    /*
    1. hidden layer 1 512 * 1 동적 할당
    2. z1 = weight layer 1 * imgBuffer[0] + bias 1 계산
    3. a1 = sigmoid * z1 계산
    4. z2 = weight layer 2 * a1 + bias 2 계산
    5. a2 = sigmoid * z2 계산
    6. z3 = weight layer 3 * a2 + bias 3 계산
    7. a3 = softmax * z3 계산
    8. 함수화 하면서 리팩토링
    */
    ////////////////////////////////////////////////////////////////////////////

    ///////////////////// Back Propagation(역전파) 연산 //////////////////////
    /*
    1. output layer의 delta(앞으로는 줄여서 d라고 호칭)값 순서대로 d0~d9까지 계산
    2. hidden layer 2 d0~d256 계산
    3. hidden layer 1 d0~d512 계산
    4. hidden layer 2 ~ output layer 간 weight 업데이트
    5. hidden layer 2 ~ hidden layer 1 간 weight 업데이트
    6. hidden layer 1 ~ input layer 간 weight 업데이트
    7. 함수화 하면서 리팩토링
    */
    ////////////////////////////////////////////////////////////////////////////

    /////////////// 나머지 9999개의 이미지 행렬화 해서 빠르게 연산 ////////////////

    ////////////////////////////////////////////////////////////////////////////

    ////////////////////// 테스트 데이터셋 이용해서 테스트 ///////////////////////

    ////////////////////////////// 메모리 해제 //////////////////////////////////
    for (int i = 0; i < batchSize; i++)
        free(imgBuffer[i]);
    free(imgBuffer);

    for (int i = 0; i < batchSize; i++)
        free(labelBuffer[i]);
    free(labelBuffer);

    fclose(img);
    fclose(label);
    ////////////////////////////////////////////////////////////////////////////

    system("pause");
}