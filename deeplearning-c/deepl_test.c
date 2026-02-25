#include <stdio.h>  // 입출력 헤더 파일
#include <stdlib.h> // 유틸리티 함수 헤더 파일
#include <math.h>   // 수학 관련 함수 헤더 파일
#include <time.h>   // 시간 관련 함수

#define ANSI_RESET "\033[0m"
#define BATCH_SIZE 32        // batch 크기
#define EPOCH_SIZE 20        // epoch 크기
#define LEARNING_RATE 0.001f // 학습률 크기
#define BETA1 0.9f
#define BETA2 0.999f
#define epsilon 1e-8f
#define TINY_NUM 1e-35f

/*
Deep Learning Information

input layer : 784 pixels
hidden layer1 : 128 neurons
    - active function : ReLU
hidden layer2 : 64 neurons
    - active function : ReLU
output layer : 10 newrons
    - active function : softmax
    - cost function : cross entropy

optimizer algorithm : Adam Optimizer
*/

typedef struct
{
    int inputSize;
    int outputSize;
    float **w;      
    float *b;       
    float **m, **v; 
    float *mb, *vb; 
} DenseLayer;

// ========================== 출력용 유틸리티 함수 ==========================
void printImagePixels(const char *name, float **inputLayer, int batchIndex, int width, int height)
{
    printf("\n=== %s (Batch Index: %d, %dx%d) ===\n", name, batchIndex, width, height);
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int pixel = (int)(inputLayer[batchIndex][y * width + x] * 255.0f);
            if (pixel == 0) printf("  . "); 
            else printf("%3d ", pixel);
        }
        printf("\n");
    }
    printf("==================================================\n\n");
}
// =======================================================================

int getArgmax(float *arr, int size)
{
    int maxIdx = 0; float maxVal = arr[0];
    for (int i = 1; i < size; i++) if (arr[i] > maxVal) { maxVal = arr[i]; maxIdx = i; }
    return maxIdx;
}

int getArgmaxChar(unsigned char *arr, int size)
{
    for (int i = 0; i < size; i++) if (arr[i] == 1) return i;
    return 0;
}

float **createBuffer(int size)
{
    float **buffer = (float **)malloc(sizeof(float *) * BATCH_SIZE);
    for (int i = 0; i < BATCH_SIZE; i++) buffer[i] = (float *)calloc(size, sizeof(float));
    return buffer;
}

unsigned char **createCharBuffer(int size)
{
    unsigned char **buffer = (unsigned char **)malloc(sizeof(unsigned char *) * BATCH_SIZE);
    for (int i = 0; i < BATCH_SIZE; i++) buffer[i] = (unsigned char *)calloc(size, sizeof(unsigned char));
    return buffer;
}

void memoryFree(float **x, int n) { for (int i = 0; i < n; i++) free(x[i]); free(x); }
void memoryFreeChar(unsigned char **x, int n) { for (int i = 0; i < n; i++) free(x[i]); free(x); }

unsigned char **allImgData(int size, int count, FILE *f)
{
    unsigned char **imgBuffer = (unsigned char **)malloc(sizeof(unsigned char *) * count);
    for (int i = 0; i < count; i++) imgBuffer[i] = (unsigned char *)malloc(sizeof(unsigned char) * size);
    for (int i = 0; i < count; i++) fread(imgBuffer[i], sizeof(unsigned char), size, f);
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

void preprocessImgData(int size, int offset, unsigned char **imgData, float **out)
{
    for (int i = 0; i < BATCH_SIZE; i++)
        for (int j = 0; j < size; j++) out[i][j] = (float)imgData[offset + i][j] / 255.0f;
}

void preprocessLabelData(int oneHotSize, int offset, unsigned char **allLabels, unsigned char **out)
{
    for (int i = 0; i < BATCH_SIZE; i++)
    {
        unsigned char correctLabel = allLabels[offset + i][0];
        for (int j = 0; j < oneHotSize; j++) out[i][j] = (j == correctLabel) ? 1 : 0;
    }
}

int reverseInt(int i)
{
    unsigned int c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

DenseLayer *createLayer(int input, int output)
{
    DenseLayer *layer = (DenseLayer *)malloc(sizeof(DenseLayer));
    layer->inputSize = input; layer->outputSize = output;

    layer->w = (float **)malloc(sizeof(float *) * output);
    float limit = sqrtf(6.0f / (float)input); // He 초기화
    for (int i = 0; i < output; i++)
    {
        layer->w[i] = (float *)malloc(sizeof(float) * input);
        for (int j = 0; j < input; j++) layer->w[i][j] = ((float)rand() / RAND_MAX) * (2.0f * limit) - limit;
    }

    layer->b = (float *)malloc(sizeof(float) * output);
    for (int i = 0; i < output; i++) layer->b[i] = ((float)rand() / RAND_MAX) * (2.0f * limit) - limit;

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

void freeLayer(DenseLayer *layer)
{
    memoryFree(layer->w, layer->outputSize); memoryFree(layer->m, layer->outputSize); memoryFree(layer->v, layer->outputSize);
    free(layer->b); free(layer->mb); free(layer->vb); free(layer);
}

float ReLU(float n) { return (n > 0) ? n : 0.0f; }
float ReLUPrime(float n) { return (n > 0) ? 1.0f : 0.0f; }

void softmax(float **z, int size, float **out)
{
    for (int k = 0; k < BATCH_SIZE; k++)
    {
        float max = z[k][0];
        for (int i = 1; i < size; i++) if (z[k][i] > max) max = z[k][i];

        float sum = 0.0f;
        for (int i = 0; i < size; i++) { out[k][i] = expf(z[k][i] - max); sum += out[k][i]; }
        for (int i = 0; i < size; i++) out[k][i] /= sum;
    }
}

void activate(float **z, int size, float (*func)(float), float **out)
{
    for (int k = 0; k < BATCH_SIZE; k++)
        for (int i = 0; i < size; i++) out[k][i] = func(z[k][i]);
}

void linear(DenseLayer *layer, float **x, float **out)
{
#pragma omp parallel for
    for (int k = 0; k < BATCH_SIZE; k++)
    {
        for (int i = 0; i < layer->outputSize; i++)
        {
            out[k][i] = layer->b[i];
            for (int j = 0; j < layer->inputSize; j++) out[k][i] += layer->w[i][j] * x[k][j];
        }
    }
}

void createOutputDelta(unsigned char **y, float **a, int size, float **out)
{
    for (int k = 0; k < BATCH_SIZE; k++)
        for (int i = 0; i < size; i++) out[k][i] = a[k][i] - (float)y[k][i];
}

void createHiddenDelta(float **z, float **nextDelta, DenseLayer *nextLayer, int currSize, float (*func)(float), float **out)
{
    for (int k = 0; k < BATCH_SIZE; k++)
    {
        for (int i = 0; i < currSize; i++)
        {
            float sum = 0.0f;
            for (int j = 0; j < nextLayer->outputSize; j++) sum += nextLayer->w[j][i] * nextDelta[k][j];
            out[k][i] = sum * func(z[k][i]);
        }
    }
}

float Adam(float dL_dw, float w, float alpha, float *m, float *v, float fix1, float fix2)
{
    *m = BETA1 * (*m) + (1.0f - BETA1) * dL_dw;
    *v = BETA2 * (*v) + (1.0f - BETA2) * (dL_dw * dL_dw);
    if (*v < TINY_NUM) *v = TINY_NUM;
    return w - alpha * (*m * fix1) / (sqrtf(*v * fix2) + epsilon);
}

void backpropagation(DenseLayer *layer, float **x, float **delta, float learningRate, int t)
{
    float fix1 = 1.0f / (1.0f - powf(BETA1, t));
    float fix2 = 1.0f / (1.0f - powf(BETA2, t));

    for (int j = 0; j < layer->outputSize; j++)
    {
        for (int i = 0; i < layer->inputSize; i++)
        {
            float grad_sum = 0.0f;
            for (int k = 0; k < BATCH_SIZE; k++) grad_sum += delta[k][j] * x[k][i];
            layer->w[j][i] = Adam(grad_sum / (float)BATCH_SIZE, layer->w[j][i], learningRate, &layer->m[j][i], &layer->v[j][i], fix1, fix2);
        }
    }

    for (int i = 0; i < layer->outputSize; i++)
    {
        float sum_delta = 0.0f;
        for (int k = 0; k < BATCH_SIZE; k++) sum_delta += delta[k][i];
        layer->b[i] = Adam(sum_delta / (float)BATCH_SIZE, layer->b[i], learningRate, &layer->mb[i], &layer->vb[i], fix1, fix2);
    }
}

float crossEntropy(float **predict, unsigned char **target)
{
    float totalLoss = 0;
    for (int i = 0; i < BATCH_SIZE; i++)
        for (int j = 0; j < 10; j++)
            if (target[i][j] == 1) totalLoss -= logf(predict[i][j] + 1e-9f);
    return totalLoss / BATCH_SIZE;
}

int main()
{
    srand(time(NULL));
    FILE *img = fopen("train-images.idx3-ubyte", "rb"), *label = fopen("train-labels.idx1-ubyte", "rb");

    int t1, t2, imgCount, labelCount, imgWidth, imgHeight;
    fread(&t1, sizeof(int), 1, img); fread(&t2, sizeof(int), 1, label);
    fread(&imgCount, sizeof(int), 1, img); imgCount = reverseInt(imgCount);
    fread(&labelCount, sizeof(int), 1, label); labelCount = reverseInt(labelCount);
    fread(&imgWidth, sizeof(int), 1, img); imgWidth = reverseInt(imgWidth);
    fread(&imgHeight, sizeof(int), 1, img); imgHeight = reverseInt(imgHeight);

    int imgSize = imgWidth * imgHeight, labelSize = 10;
    unsigned char **allImage = allImgData(imgSize, imgCount, img);
    unsigned char **allLabels = allLabelData(labelCount, label);

    int hidden1 = 128, hidden2 = 64, output = 10;
    DenseLayer *layer1 = createLayer(imgSize, hidden1);
    DenseLayer *layer2 = createLayer(hidden1, hidden2);
    DenseLayer *layer3 = createLayer(hidden2, output);

    float **inputBuffer = createBuffer(imgSize);
    unsigned char **labelBuffer = createCharBuffer(labelSize);
    float **z1Buf = createBuffer(hidden1), **a1Buf = createBuffer(hidden1), **d1Buf = createBuffer(hidden1);
    float **z2Buf = createBuffer(hidden2), **a2Buf = createBuffer(hidden2), **d2Buf = createBuffer(hidden2);
    float **z3Buf = createBuffer(output), **a3Buf = createBuffer(output), **d3Buf = createBuffer(output);

    int totalStep = 1;

    for (int e = 0; e < EPOCH_SIZE; e++)
    {
        printf("------EPOCH %d------\n", e + 1);

        for (int i = 0; i < imgCount / BATCH_SIZE; i++)
        {
            float offset = i * BATCH_SIZE;
            preprocessImgData(imgSize, offset, allImage, inputBuffer);
            preprocessLabelData(labelSize, offset, allLabels, labelBuffer);

            // 순전파 연산
            linear(layer1, inputBuffer, z1Buf); activate(z1Buf, hidden1, ReLU, a1Buf);
            linear(layer2, a1Buf, z2Buf); activate(z2Buf, hidden2, ReLU, a2Buf);
            linear(layer3, a2Buf, z3Buf); softmax(z3Buf, output, a3Buf);

            // 역전파 오차 연산
            createOutputDelta(labelBuffer, a3Buf, output, d3Buf);
            createHiddenDelta(z2Buf, d3Buf, layer3, hidden2, ReLUPrime, d2Buf);
            createHiddenDelta(z1Buf, d2Buf, layer2, hidden1, ReLUPrime, d1Buf);

            // =========================================================================================
            // [발표 자료용 출력] Adam Optimizer 변수 검증 및 추적 
            // =========================================================================================
            if (e == 0 && i == 0)
            {
                printImagePixels("학습에 사용된 1번째 이미지 픽셀 구조", inputBuffer, 0, imgWidth, imgHeight);

                int src_nodes[3] = {0,};
                int dst_nodes[3] = {0, 1, 2}; 
                int found = 0;
                for (int p = 0; p < imgSize && found < 3; p++) {
                    if (inputBuffer[0][p] > 0.0f) { src_nodes[found] = p; found++; }
                }
                while (found < 3) { src_nodes[found] = found; found++; }

                printf("\n==========================================================================================\n");
                printf("  [발표 자료] Adam Optimizer Step 1 수학적 무결성 상세 검증 (3x3 Subgraph)\n");
                printf("==========================================================================================\n\n");

                printf(">>> [Adam 수식 변수 추적]\n");
                printf("  * t(Step) = %d\n", totalStep);
                printf("  * alpha(학습률) = %.4f | Beta1 = %.3f | Beta2 = %.3f\n\n", LEARNING_RATE, BETA1, BETA2);

                // 고정밀도 Adam 연산 출력 매크로 (수치적 무결성 검증용)
                #define PRINT_ADAM_DETAILED(LAYER, SRC_NAME, DST_NAME, DELTA, PREV_A, SRC_NODES) \
                for (int d = 0; d < 3; d++) { \
                    for (int s = 0; s < 3; s++) { \
                        int in_idx = SRC_NODES[s]; \
                        int out_idx = dst_nodes[d]; \
                        float grad_w = 0.0f; \
                        for (int b = 0; b < BATCH_SIZE; b++) grad_w += DELTA[b][out_idx] * PREV_A[b][in_idx]; \
                        grad_w /= (float)BATCH_SIZE; \
                        float w_old = LAYER->w[out_idx][in_idx]; \
                        float m_old = LAYER->m[out_idx][in_idx], v_old = LAYER->v[out_idx][in_idx]; \
                        float m_new = BETA1 * m_old + (1.0f - BETA1) * grad_w; \
                        float v_tmp = BETA2 * v_old + (1.0f - BETA2) * (grad_w * grad_w); \
                        float v_new = (v_tmp < TINY_NUM) ? TINY_NUM : v_tmp; \
                        float fix1_val = 1.0f - powf(BETA1, totalStep), fix2_val = 1.0f - powf(BETA2, totalStep); \
                        float fix1 = 1.0f / fix1_val, fix2 = 1.0f / fix2_val; \
                        float m_hat = m_new * fix1, v_hat = v_new * fix2; \
                        float update_term = LEARNING_RATE * m_hat / (sqrtf(v_hat) + epsilon); \
                        float predicted_w = w_old - update_term; \
                        printf("  [%s[%d] -> %s[%d]]\n", SRC_NAME, in_idx, DST_NAME, out_idx); \
                        printf("    1) 기울기(g_t) : %.8f | 기존 가중치(w_old) : %.8f\n", grad_w, w_old); \
                        printf("    2) m_t = %.3f*%.1f + %.3f*(%.8f) = %.8e\n", BETA1, m_old, 1.0f-BETA1, grad_w, m_new); \
                        printf("       => m_hat (m_t / %.3f) = %.8e\n", fix1_val, m_hat); \
                        printf("    3) v_t = %.3f*%.1f + %.3f*(%.8f)^2 = %.8e\n", BETA2, v_old, 1.0f-BETA2, grad_w, v_new); \
                        printf("       => v_hat (v_t / %.3f) = %.8e\n", fix2_val, v_hat); \
                        printf("    4) w_new = %.8f - %.4f * (%.8e / (sqrt(%.8e) + %.0e))\n", w_old, LEARNING_RATE, m_hat, v_hat, epsilon); \
                        printf("       최종 업데이트 값(w_new) = %.8f\n\n", predicted_w); \
                    } \
                }

                printf("  [Layer 1 가중치 업데이트 (Input -> Hidden 1)]\n");
                PRINT_ADAM_DETAILED(layer1, "input layer", "hidden layer 1", d1Buf, inputBuffer, src_nodes);

                printf("  [Layer 2 가중치 업데이트 (Hidden 1 -> Hidden 2)]\n");
                PRINT_ADAM_DETAILED(layer2, "hidden layer 1", "hidden layer 2", d2Buf, a1Buf, dst_nodes);

                printf("  [Layer 3 가중치 업데이트 (Hidden 2 -> Output)]\n");
                PRINT_ADAM_DETAILED(layer3, "hidden layer 2", "output layer", d3Buf, a2Buf, dst_nodes);
                printf("\n==========================================================================================\n\n");
            }

            // =========================================================================================
            // 실제 메모리의 가중치를 업데이트 시키는 Backpropagation 함수 호출
            backpropagation(layer3, a2Buf, d3Buf, LEARNING_RATE, totalStep);
            backpropagation(layer2, a1Buf, d2Buf, LEARNING_RATE, totalStep);
            backpropagation(layer1, inputBuffer, d1Buf, LEARNING_RATE, totalStep);
            // =========================================================================================

            if (i % 100 == 0) {
                float currentLoss = crossEntropy(a3Buf, labelBuffer);
                printf("Epoch [%2d/%2d] Batch [%4d/%4d] Loss: %.4f\n", e + 1, EPOCH_SIZE, i, imgCount / BATCH_SIZE, currentLoss);
            }
            totalStep++;
        }
    }

    ////////////////////////////// 메모리 해제 //////////////////////////////////
    memoryFree(inputBuffer, BATCH_SIZE); memoryFreeChar(labelBuffer, BATCH_SIZE);
    memoryFree(z1Buf, BATCH_SIZE); memoryFree(a1Buf, BATCH_SIZE); memoryFree(d1Buf, BATCH_SIZE);
    memoryFree(z2Buf, BATCH_SIZE); memoryFree(a2Buf, BATCH_SIZE); memoryFree(d2Buf, BATCH_SIZE);
    memoryFree(z3Buf, BATCH_SIZE); memoryFree(a3Buf, BATCH_SIZE); memoryFree(d3Buf, BATCH_SIZE);

    freeLayer(layer1); freeLayer(layer2); freeLayer(layer3);
    fclose(img); fclose(label);
    
    system("pause");
}