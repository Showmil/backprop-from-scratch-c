#include <stdio.h>  // 입출력 헤더 파일
#include <stdlib.h> // 유틸리티 함수 헤더 파일
#include <math.h>   // 수학 관련 함수 헤더 파일
#include <time.h>   // 시간 관련 함수
#include "raylib.h" // 시각화 관련 Raylib 헤더 추가

#define ANSI_RESET "\033[0m"
#define BATCH_SIZE 32        // batch 크기
#define EPOCH_SIZE 1        // epoch 크기
#define LEARNING_RATE 0.001f // 학습률 크기
#define BETA1 0.9f
#define BETA2 0.999f
#define epsilon 1e-8f
#define TINY_NUM 1e-35f

// 실시간 그래프를 위한 버퍼 크기 설정
#define GRAPH_WIDTH 400

// Layer 구조체
typedef struct
{
    int inputSize;
    int outputSize;
    float **w;      // 가중치
    float *b;       // 편향
    float **m, **v; // Adam 가중치 파라미터
    float *mb, *vb; // Adam 편향 파라미터
} DenseLayer;

// 다크모드용 가중치 -> 색상 변환 함수 (Red & Cyan)
Color GetWeightColor(float weight, float max_val)
{
    float val = weight / max_val;
    if (val > 1.0f)
        val = 1.0f;
    if (val < -1.0f)
        val = -1.0f;

    if (val > 0)
    {
        return (Color){(unsigned char)(val * 255), 0, 50, 255}; // 양수: 강렬한 빨강/핑크
    }
    else
    {
        return (Color){0, (unsigned char)(-val * 200), (unsigned char)(-val * 255), 255}; // 음수: 시안/블루
    }
}

// 가장 높은 값 인덱스 찾는 함수 (예측값)
int getArgmax(float *arr, int size)
{
    int maxIdx = 0;
    float maxVal = arr[0];
    for (int i = 1; i < size; i++)
    {
        if (arr[i] > maxVal)
        {
            maxVal = arr[i];
            maxIdx = i;
        }
    }
    return maxIdx;
}

//  One-hot 인코딩에서 정답 인덱스 찾는 함수
int getArgmaxChar(unsigned char *arr, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (arr[i] == 1)
            return i;
    }
    return 0;
}

// float 2차원 버퍼 생성 함수
float **createBuffer(int size)
{
    float **buffer = (float **)malloc(sizeof(float *) * BATCH_SIZE);
    for (int i = 0; i < BATCH_SIZE; i++)
    {
        buffer[i] = (float *)calloc(size, sizeof(float)); // 0으로 초기화
    }
    return buffer;
}

// unsigned char 2차원 버퍼 생성 함수
unsigned char **createCharBuffer(int size)
{
    unsigned char **buffer = (unsigned char **)malloc(sizeof(unsigned char *) * BATCH_SIZE);
    for (int i = 0; i < BATCH_SIZE; i++)
    {
        buffer[i] = (unsigned char *)calloc(size, sizeof(unsigned char));
    }
    return buffer;
}

// 메모리 해제 함수
void memoryFree(float **x, int n)
{
    for (int i = 0; i < n; i++)
        free(x[i]);
    free(x);
}

// char 메모리 해제 함수
void memoryFreeChar(unsigned char **x, int n)
{
    for (int i = 0; i < n; i++)
        free(x[i]);
    free(x);
}

// 데이터 로드 함수
unsigned char **allImgData(int size, int count, FILE *f)
{
    unsigned char **imgBuffer = (unsigned char **)malloc(sizeof(unsigned char *) * count);
    for (int i = 0; i < count; i++)
        imgBuffer[i] = (unsigned char *)malloc(sizeof(unsigned char) * size);
    for (int i = 0; i < count; i++)
        fread(imgBuffer[i], sizeof(unsigned char), size, f);
    return imgBuffer;
}

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

// 이미지 데이터 전처리
void preprocessImgData(int size, int offset, unsigned char **imgData, float **out)
{
    for (int i = 0; i < BATCH_SIZE; i++)
    {
        for (int j = 0; j < size; j++)
        {
            out[i][j] = (float)imgData[offset + i][j] / 255.0f;
        }
    }
}

// 라벨 데이터 전처리
void preprocessLabelData(int oneHotSize, int offset, unsigned char **allLabels, unsigned char **out)
{
    for (int i = 0; i < BATCH_SIZE; i++)
    {
        unsigned char correctLabel = allLabels[offset + i][0];
        for (int j = 0; j < oneHotSize; j++)
        {
            out[i][j] = (j == correctLabel) ? 1 : 0;
        }
    }
}

// 4바이트 뒤집기 함수
int reverseInt(int i)
{
    unsigned int c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

// Layer 생성 및 초기화
DenseLayer *createLayer(int input, int output)
{
    DenseLayer *layer = (DenseLayer *)malloc(sizeof(DenseLayer));
    layer->inputSize = input;
    layer->outputSize = output;

    layer->w = (float **)malloc(sizeof(float *) * output);
    float limit = sqrtf(6.0f / (float)input); // He 초기화
    for (int i = 0; i < output; i++)
    {
        layer->w[i] = (float *)malloc(sizeof(float) * input);
        for (int j = 0; j < input; j++)
            layer->w[i][j] = ((float)rand() / RAND_MAX) * (2.0f * limit) - limit;
    }

    layer->b = (float *)malloc(sizeof(float) * output);
    for (int i = 0; i < output; i++)
        layer->b[i] = ((float)rand() / RAND_MAX) * (2.0f * limit) - limit;

    layer->m = (float **)malloc(sizeof(float *) * output);
    layer->v = (float **)malloc(sizeof(float *) * output);
    for (int i = 0; i < output; i++)
    {
        layer->m[i] = (float *)calloc(input, sizeof(float));
        layer->v[i] = (float *)calloc(input, sizeof(float));
    }

    layer->mb = (float *)calloc(output, sizeof(float));
    layer->vb = (float *)calloc(output, sizeof(float));
    return layer;
}

// Layer 메모리 해제 함수
void freeLayer(DenseLayer *layer)
{
    memoryFree(layer->w, layer->outputSize);
    memoryFree(layer->m, layer->outputSize);
    memoryFree(layer->v, layer->outputSize);
    free(layer->b);
    free(layer->mb);
    free(layer->vb);
    free(layer);
}

// sigmoid 함수
float sigmoid(float n) { return 1.0f / (1.0f + expf(-n)); }
float sigmoidPrime(float n) { return sigmoid(n) * (1.0f - sigmoid(n)); }

// ReLU 함수
float ReLU(float n) { return (n > 0) ? n : 0.0f; }
float ReLUPrime(float n) { return (n > 0) ? 1.0f : 0.0f; }

// Softmax 함수
void softmax(float **z, int size, float **out)
{
    for (int k = 0; k < BATCH_SIZE; k++)
    {
        float max = z[k][0];
        for (int i = 1; i < size; i++)
            if (z[k][i] > max)
                max = z[k][i];

        float sum = 0.0f;
        for (int i = 0; i < size; i++)
        {
            out[k][i] = expf(z[k][i] - max);
            sum += out[k][i];
        }
        for (int i = 0; i < size; i++)
            out[k][i] /= sum;
    }
}

// 활성화 함수
void activate(float **z, int size, float (*func)(float), float **out)
{
    for (int k = 0; k < BATCH_SIZE; k++)
    {
        for (int i = 0; i < size; i++)
            out[k][i] = func(z[k][i]);
    }
}

// 순전파 함수
void linear(DenseLayer *layer, float **x, float **out)
{
#pragma omp parallel for
    for (int k = 0; k < BATCH_SIZE; k++)
    {
        for (int i = 0; i < layer->outputSize; i++)
        {
            out[k][i] = layer->b[i]; // 편향
            for (int j = 0; j < layer->inputSize; j++)
            {
                out[k][i] += layer->w[i][j] * x[k][j]; // 가중치 연산
            }
        }
    }
}

// 출력층 오차(Delta) 계산
void createOutputDelta(unsigned char **y, float **a, int size, float **out)
{
    for (int k = 0; k < BATCH_SIZE; k++)
    {
        for (int i = 0; i < size; i++)
        {
            out[k][i] = a[k][i] - (float)y[k][i];
        }
    }
}

// 은닉층 오차(Delta) 계산
void createHiddenDelta(float **z, float **nextDelta, DenseLayer *nextLayer, int currSize, float (*func)(float), float **out)
{
    for (int k = 0; k < BATCH_SIZE; k++)
    {
        for (int i = 0; i < currSize; i++)
        {
            float sum = 0.0f;
            for (int j = 0; j < nextLayer->outputSize; j++)
            {
                sum += nextLayer->w[j][i] * nextDelta[k][j];
            }
            out[k][i] = sum * func(z[k][i]);
        }
    }
}

// Adam Optimizer
float Adam(float dL_dw, float w, float alpha, float *m, float *v, float fix1, float fix2)
{
    *m = BETA1 * (*m) + (1.0f - BETA1) * dL_dw;
    *v = BETA2 * (*v) + (1.0f - BETA2) * (dL_dw * dL_dw);

    // v가 0에 너무 가까우면 강제로 아주 작은 값을 더해줌
    if (*v < TINY_NUM)
        *v = TINY_NUM;

    float m_hat = *m * fix1;
    float v_hat = *v * fix2;
    return w - alpha * m_hat / (sqrtf(v_hat) + epsilon);
}

// Backpropagation
void backpropagation(DenseLayer *layer, float **x, float **delta, float learningRate, int t)
{
    float fix1 = 1.0f / (1.0f - powf(BETA1, t));
    float fix2 = 1.0f / (1.0f - powf(BETA2, t));

    // 가중치(w) 업데이트
    for (int j = 0; j < layer->outputSize; j++)
    {
        for (int i = 0; i < layer->inputSize; i++)
        {
            float grad_sum = 0.0f;
            for (int k = 0; k < BATCH_SIZE; k++)
                grad_sum += delta[k][j] * x[k][i];

            float dL_dw = grad_sum / (float)BATCH_SIZE;
            layer->w[j][i] = Adam(dL_dw, layer->w[j][i], learningRate, &layer->m[j][i], &layer->v[j][i], fix1, fix2);
        }
    }

    // 편향(b) 업데이트
    for (int i = 0; i < layer->outputSize; i++)
    {
        float sum_delta = 0.0f;
        for (int k = 0; k < BATCH_SIZE; k++)
            sum_delta += delta[k][i];

        float dL_db = sum_delta / (float)BATCH_SIZE;
        layer->b[i] = Adam(dL_db, layer->b[i], learningRate, &layer->mb[i], &layer->vb[i], fix1, fix2);
    }
}

// Cross-Entropy 손실 함수
float crossEntropy(float **predict, unsigned char **target)
{
    float totalLoss = 0;
    for (int i = 0; i < BATCH_SIZE; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            if (target[i][j] == 1)
                totalLoss -= logf(predict[i][j] + 1e-9f);
        }
    }
    return totalLoss / BATCH_SIZE;
}

int main()
{
    srand(time(NULL));
    clock_t startTime = clock(); // 시간 측정 시작

    // 데이터 파일 열기
    FILE *img = fopen("train-images.idx3-ubyte", "rb");
    FILE *label = fopen("train-labels.idx1-ubyte", "rb");

    // 데이터 헤더 정보 읽기
    int dataMagicNumber, labelMagicNumber, imgCount, labelCount, imgWidth, imgHeight;
    fread(&dataMagicNumber, sizeof(int), 1, img);
    dataMagicNumber = reverseInt(dataMagicNumber);
    fread(&labelMagicNumber, sizeof(int), 1, label);
    labelMagicNumber = reverseInt(labelMagicNumber);
    fread(&imgCount, sizeof(int), 1, img);
    imgCount = reverseInt(imgCount);
    fread(&labelCount, sizeof(int), 1, label);
    labelCount = reverseInt(labelCount);
    fread(&imgWidth, sizeof(int), 1, img);
    imgWidth = reverseInt(imgWidth);
    fread(&imgHeight, sizeof(int), 1, img);
    imgHeight = reverseInt(imgHeight);

    int imgSize = imgWidth * imgHeight;
    int labelSize = 10;

    unsigned char **allImage = allImgData(imgSize, imgCount, img);
    unsigned char **allLabels = allLabelData(labelCount, label);

    // Layer 생성 및 초기화
    int hidden1 = 128, hidden2 = 64, output = 10;
    DenseLayer *layer1 = createLayer(imgSize, hidden1);
    DenseLayer *layer2 = createLayer(hidden1, hidden2);
    DenseLayer *layer3 = createLayer(hidden2, output);

    // --------------------- 메모리 버퍼 생성 ---------------------
    float **inputBuffer = createBuffer(imgSize);
    unsigned char **labelBuffer = createCharBuffer(labelSize);

    float **z1Buf = createBuffer(hidden1);
    float **a1Buf = createBuffer(hidden1);
    float **d1Buf = createBuffer(hidden1);
    float **z2Buf = createBuffer(hidden2);
    float **a2Buf = createBuffer(hidden2);
    float **d2Buf = createBuffer(hidden2);
    float **z3Buf = createBuffer(output);
    float **a3Buf = createBuffer(output);
    float **d3Buf = createBuffer(output);

    int totalStep = 1;
    float finalTrainLoss = 0.0f;
    float finalTrainAcc = 0.0f;

    // =====================================================================
    // Raylib 실시간 렌더링 루프 시작
    // =====================================================================
    float currentLoss = 0.0f;
    int e = 0;
    int i = 0;
    int batchCount = imgCount / BATCH_SIZE;

    // 실시간 그래프 데이터 버퍼
    float lossHistory[GRAPH_WIDTH] = {0};
    int historyIndex = 0;

    // 일시정지를 위한 상태 변수 추가
    bool isPaused = false;

    // 화면 비율 와이드하게 확장
    const int screenWidth = 1400;
    const int screenHeight = 800;
    InitWindow(screenWidth, screenHeight, "Deep Learning Backprop Visualizer - Master Edition");
    SetTargetFPS(60);

    while (!WindowShouldClose())
    {
        // 스페이스바를 누르면 일시정지 상태 전환
        if (IsKeyPressed(KEY_SPACE)) isPaused = !isPaused;

        // --- 1. 데이터 연산 파트 (일시정지가 아닐 때만 실행) ---
        if (!isPaused && e < EPOCH_SIZE)
        {
            float offset = i * BATCH_SIZE;
            preprocessImgData(imgSize, offset, allImage, inputBuffer);
            preprocessLabelData(labelSize, offset, allLabels, labelBuffer);

            // Forward
            linear(layer1, inputBuffer, z1Buf);
            activate(z1Buf, hidden1, ReLU, a1Buf);
            linear(layer2, a1Buf, z2Buf);
            activate(z2Buf, hidden2, ReLU, a2Buf);
            linear(layer3, a2Buf, z3Buf);
            softmax(z3Buf, output, a3Buf);

            currentLoss = crossEntropy(a3Buf, labelBuffer);

            // 그래프 데이터 수집
            if (i % 5 == 0 && historyIndex < GRAPH_WIDTH)
            {
                lossHistory[historyIndex] = currentLoss;
                historyIndex++;
            }

            // Backward
            createOutputDelta(labelBuffer, a3Buf, output, d3Buf);
            createHiddenDelta(z2Buf, d3Buf, layer3, hidden2, ReLUPrime, d2Buf);
            createHiddenDelta(z1Buf, d2Buf, layer2, hidden1, ReLUPrime, d1Buf);

            backpropagation(layer3, a2Buf, d3Buf, LEARNING_RATE, totalStep);
            backpropagation(layer2, a1Buf, d2Buf, LEARNING_RATE, totalStep);
            backpropagation(layer1, inputBuffer, d1Buf, LEARNING_RATE, totalStep);

            totalStep++;
            i++;

            if (i >= batchCount)
            {
                i = 0;
                e++;
                historyIndex = 0; // 새 에폭시 그래프 초기화
            }
        }

        // --- 2. 화면 그리기 파트 ---
        BeginDrawing();
        ClearBackground((Color){15, 15, 20, 255}); // 다크 모드 배경

        // [상단] 텍스트 정보
        DrawText(TextFormat("EPOCH: [%d / %d]", e + 1, EPOCH_SIZE), 30, 30, 25, GREEN);
        DrawText(TextFormat("BATCH: [%d / %d]", i, batchCount), 30, 65, 20, LIGHTGRAY);
        DrawText(TextFormat("LOSS : %.4f", currentLoss), 30, 100, 20, currentLoss < 0.2f ? GREEN : RED);

        // 일시정지 상태 화면 표시
        if (isPaused) {
            DrawText("PAUSED (Press SPACE to Resume)", 250, 30, 25, YELLOW);
        }

        // [상단 우측] 실시간 Loss 그래프
        DrawText("Learning Curve (Loss)", screenWidth - GRAPH_WIDTH - 50, 30, 20, SKYBLUE);
        int graphX = screenWidth - GRAPH_WIDTH - 50;
        int graphY = 60;
        int graphHeight = 100;
        DrawRectangleLines(graphX, graphY, GRAPH_WIDTH, graphHeight, DARKGRAY);

        for (int h = 1; h < historyIndex; h++)
        {
            float scaledY1 = (lossHistory[h - 1] / 3.0f) * graphHeight;
            float scaledY2 = (lossHistory[h] / 3.0f) * graphHeight;

            if (scaledY1 > graphHeight) scaledY1 = graphHeight;
            if (scaledY2 > graphHeight) scaledY2 = graphHeight;

            DrawLine(graphX + h - 1, graphY + graphHeight - (int)scaledY1,
                     graphX + h, graphY + graphHeight - (int)scaledY2, GREEN);
        }

        // UI 구역 분할 프레임 (파이프라인 시각화)
        int panelY = 160;
        DrawRectangleLinesEx((Rectangle){ 20, panelY, 180, 260 }, 2, DARKGRAY);
        DrawText("->", 210, panelY + 100, 30, DARKGRAY); // 연결 화살표
        
        DrawRectangleLinesEx((Rectangle){ 240, panelY, 1100, 260 }, 2, DARKGRAY);
        
        DrawRectangleLinesEx((Rectangle){ 20, 440, 1320, 330 }, 2, DARKGRAY);

        // ==========================================================
        // 1. 현재 입력 이미지 (Input Layer)
        // ==========================================================
        DrawText("1. Input Image", 30, 170, 20, SKYBLUE);
        int imgStartX = 30;
        int imgStartY = 210;
        int imgPixelSize = 5; // 28*5 = 140

        for (int y = 0; y < 28; y++)
        {
            for (int x = 0; x < 28; x++)
            {
                float val = inputBuffer[0][y * 28 + x];
                Color c = (Color){(unsigned char)(val * 255), (unsigned char)(val * 255), (unsigned char)(val * 255), 255};
                DrawRectangle(imgStartX + x * imgPixelSize, imgStartY + y * imgPixelSize, imgPixelSize, imgPixelSize, c);
            }
        }

        // ==========================================================
        // 2. Layer 1 가중치 특징(Feature) 지도
        // ==========================================================
        DrawText("2. Hidden Layer 1 (Feature Extractors)", 250, 170, 20, SKYBLUE);
        int l1StartX = 250;
        int l1StartY = 210;
        int l1PixelSize = 3; // 28*3 = 84
        int l1Spacing = 20;

        for (int n = 0; n < 10; n++)
        {
            int curX = l1StartX + n * (28 * l1PixelSize + l1Spacing);
            DrawText(TextFormat("N%d", n + 1), curX + 5, l1StartY - 25, 15, LIGHTGRAY);

            for (int wy = 0; wy < 28; wy++)
            {
                for (int wx = 0; wx < 28; wx++)
                {
                    float weight_val = layer1->w[n][wy * 28 + wx];
                    Color c = GetWeightColor(weight_val, 0.2f); // 민감도 0.2
                    DrawRectangle(curX + wx * l1PixelSize, l1StartY + wy * l1PixelSize, l1PixelSize, l1PixelSize, c);
                }
            }

            // ReLU 게이트 역할 시각화 (활성화 값 표시)
            float actValue = a1Buf[0][n];
            Color actColor = (actValue > 0.0f) ? YELLOW : DARKGRAY; // 0이면 회색(죽음), 그 이상이면 노란색
            DrawText(TextFormat("Act: %.2f", actValue), curX + 5, l1StartY + (28 * l1PixelSize) + 10, 15, actColor);
        }

        // ==========================================================
        // 3. 예측 확률 (막대그래프)
        // ==========================================================
        DrawText("3. Output Probabilities (Softmax)", 30, 450, 20, SKYBLUE);
        int barStartX = 30;
        int barStartY = 750; // 바닥 기준점
        int barWidth = 70;
        int barSpacing = 50;
        int maxBarHeight = 220;

        int correctAns = getArgmaxChar(labelBuffer[0], 10);
        int predictedAns = getArgmax(a3Buf[0], 10);

        DrawText(TextFormat("TARGET : %d", correctAns), 30, 490, 25, GREEN);
        DrawText(TextFormat("PREDICT: %d", predictedAns), 250, 490, 25, (correctAns == predictedAns) ? GREEN : RED);

        for (int n = 0; n < 10; n++)
        {
            float prob = a3Buf[0][n];
            int barHeight = (int)(prob * maxBarHeight);
            int curX = barStartX + n * (barWidth + barSpacing);

            // 색상 로직: 정답은 초록, 오답인데 가장 높게 예측했으면 빨강, 나머지는 어두운 회색
            Color barColor;
            if (n == correctAns)
                barColor = GREEN;
            else if (n == predictedAns)
                barColor = RED;
            else
                barColor = DARKGRAY;

            DrawRectangle(curX, barStartY - barHeight, barWidth, barHeight, barColor);

            DrawText(TextFormat("%d", n), curX + 30, barStartY + 10, 20, LIGHTGRAY);
            DrawText(TextFormat("%d%%", (int)(prob * 100)), curX + 15, barStartY - barHeight - 25, 18, LIGHTGRAY);
        }

        EndDrawing();
    }

    CloseWindow();

    // -------------------------------------------------------------
    // TEST 및 출력 부분 
    // -------------------------------------------------------------
    FILE *testImg = fopen("t10k-images.idx3-ubyte", "rb");
    FILE *testLabel = fopen("t10k-labels.idx1-ubyte", "rb");

    int tDMN, tLMN, tIC, tLC, tIW, tIH;
    fread(&tDMN, sizeof(int), 1, testImg);
    tDMN = reverseInt(tDMN);
    fread(&tLMN, sizeof(int), 1, testLabel);
    tLMN = reverseInt(tLMN);
    fread(&tIC, sizeof(int), 1, testImg);
    tIC = reverseInt(tIC);
    fread(&tLC, sizeof(int), 1, testLabel);
    tLC = reverseInt(tLC);
    fread(&tIW, sizeof(int), 1, testImg);
    tIW = reverseInt(tIW);
    fread(&tIH, sizeof(int), 1, testImg);
    tIH = reverseInt(tIH);

    int testImgSize = tIW * tIH;
    unsigned char **testAllImage = allImgData(testImgSize, tIC, testImg);
    unsigned char **testAllLabels = allLabelData(tLC, testLabel);

    float testTotalLoss = 0.0f;
    int testCorrectCount = 0;
    int testBatches = tIC / BATCH_SIZE;

    for (int idx = 0; idx < testBatches; idx++)
    {
        float offset = idx * BATCH_SIZE;
        preprocessImgData(testImgSize, offset, testAllImage, inputBuffer);
        preprocessLabelData(labelSize, offset, testAllLabels, labelBuffer);

        linear(layer1, inputBuffer, z1Buf);
        activate(z1Buf, hidden1, ReLU, a1Buf);
        linear(layer2, a1Buf, z2Buf);
        activate(z2Buf, hidden2, ReLU, a2Buf);
        linear(layer3, a2Buf, z3Buf);
        softmax(z3Buf, output, a3Buf);

        testTotalLoss += crossEntropy(a3Buf, labelBuffer);
        for (int k = 0; k < BATCH_SIZE; k++)
        {
            if (getArgmax(a3Buf[k], output) == getArgmaxChar(labelBuffer[k], labelSize))
            {
                testCorrectCount++;
            }
        }
    }

    float finalTestLoss = testTotalLoss / testBatches;
    float finalTestAcc = (float)testCorrectCount / (testBatches * BATCH_SIZE) * 100.0f;

    clock_t endTime = clock();
    float totalTime = (float)(endTime - startTime) / CLOCKS_PER_SEC;

    printf("\n");
    printf("----------------------------------------\n");
    printf("Final Test Loss      : %.5f\n", finalTestLoss);
    printf("Final Test Accuracy  : %.2f%%\n", finalTestAcc);
    printf("========================================\n");
    printf("Total Training Time  : %.3f seconds\n", totalTime);
    printf("========================================\n");
    printf("\n");

    memoryFree(inputBuffer, BATCH_SIZE);
    memoryFreeChar(labelBuffer, BATCH_SIZE);
    memoryFree(z1Buf, BATCH_SIZE);
    memoryFree(a1Buf, BATCH_SIZE);
    memoryFree(d1Buf, BATCH_SIZE);
    memoryFree(z2Buf, BATCH_SIZE);
    memoryFree(a2Buf, BATCH_SIZE);
    memoryFree(d2Buf, BATCH_SIZE);
    memoryFree(z3Buf, BATCH_SIZE);
    memoryFree(a3Buf, BATCH_SIZE);
    memoryFree(d3Buf, BATCH_SIZE);
    freeLayer(layer1);
    freeLayer(layer2);
    freeLayer(layer3);
    fclose(img);
    fclose(label);
    fclose(testImg);
    fclose(testLabel);

    system("pause");
}