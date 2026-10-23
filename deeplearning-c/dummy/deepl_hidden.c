#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BATCH_SIZE 128
#define EPOCH_SIZE 5
#define LEARNING_RATE 0.001f
#define BETA1 0.9f
#define BETA2 0.999f
#define EPSILON 1e-8f
#define TINY_NUM 1e-35f

#define INPUT_SIZE 784
#define OUTPUT_SIZE 10
#define MAX_HIDDEN_LAYERS 4
#define MAX_EXPERIMENTS 20

typedef struct
{
    int inputSize;
    int outputSize;
    float **w;
    float *b;
    float **m, **v;
    float *mb, *vb;
} DenseLayer;

typedef struct
{
    int hiddenCount;
    int hiddenSizes[MAX_HIDDEN_LAYERS];

    float finalTrainLoss;
    float finalTrainAcc;
    float finalTestLoss;
    float finalTestAcc;
} ExperimentResult;

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

float **createBuffer(int size)
{
    float **buffer = (float **)malloc(sizeof(float *) * BATCH_SIZE);
    for (int i = 0; i < BATCH_SIZE; i++)
        buffer[i] = (float *)calloc(size, sizeof(float));
    return buffer;
}

void freeBuffer(float **x, int n)
{
    for (int i = 0; i < n; i++)
        free(x[i]);
    free(x);
}

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
    unsigned char **labelBuffer = (unsigned char **)malloc(sizeof(unsigned char *) * count);
    for (int i = 0; i < count; i++)
    {
        labelBuffer[i] = (unsigned char *)malloc(sizeof(unsigned char));
        fread(labelBuffer[i], sizeof(unsigned char), 1, f);
    }
    return labelBuffer;
}

void freeAllImgData(unsigned char **data, int count)
{
    for (int i = 0; i < count; i++)
        free(data[i]);
    free(data);
}

void freeAllLabelData(unsigned char **data, int count)
{
    for (int i = 0; i < count; i++)
        free(data[i]);
    free(data);
}

float **preprocessAllImages(unsigned char **rawImage, int count, int imgSize)
{
    float **out = (float **)malloc(sizeof(float *) * count);
    for (int i = 0; i < count; i++)
    {
        out[i] = (float *)malloc(sizeof(float) * imgSize);
        for (int j = 0; j < imgSize; j++)
            out[i][j] = (float)rawImage[i][j] / 255.0f;
    }
    return out;
}

int *preprocessAllLabels(unsigned char **rawLabels, int count)
{
    int *out = (int *)malloc(sizeof(int) * count);
    for (int i = 0; i < count; i++)
        out[i] = (int)rawLabels[i][0];
    return out;
}

void freeAllFloatData(float **data, int count)
{
    for (int i = 0; i < count; i++)
        free(data[i]);
    free(data);
}

void copyImageBatch(int offset, float **allImage, float **out)
{
    for (int i = 0; i < BATCH_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            out[i][j] = allImage[offset + i][j];
}

void copyLabelBatch(int offset, int *allLabels, int *out)
{
    for (int i = 0; i < BATCH_SIZE; i++)
        out[i] = allLabels[offset + i];
}

int reverseInt(int i)
{
    unsigned int c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

DenseLayer *createLayer(int input, int output)
{
    DenseLayer *layer = (DenseLayer *)malloc(sizeof(DenseLayer));
    layer->inputSize = input;
    layer->outputSize = output;

    layer->w = (float **)malloc(sizeof(float *) * output);
    float limit = sqrtf(6.0f / (float)input);

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

void freeLayer(DenseLayer *layer)
{
    freeBuffer(layer->w, layer->outputSize);
    freeBuffer(layer->m, layer->outputSize);
    freeBuffer(layer->v, layer->outputSize);
    free(layer->b);
    free(layer->mb);
    free(layer->vb);
    free(layer);
}

float ReLU(float n) { return (n > 0.0f) ? n : 0.0f; }
float ReLUPrime(float n) { return (n > 0.0f) ? 1.0f : 0.0f; }

void linear(DenseLayer *layer, float **x, float **out)
{
    for (int k = 0; k < BATCH_SIZE; k++)
    {
        for (int i = 0; i < layer->outputSize; i++)
        {
            float sum = layer->b[i];
            for (int j = 0; j < layer->inputSize; j++)
                sum += layer->w[i][j] * x[k][j];
            out[k][i] = sum;
        }
    }
}

void activate(float **z, int size, float (*func)(float), float **out)
{
    for (int k = 0; k < BATCH_SIZE; k++)
        for (int i = 0; i < size; i++)
            out[k][i] = func(z[k][i]);
}

void softmax(float **z, int size, float **out)
{
    for (int k = 0; k < BATCH_SIZE; k++)
    {
        float maxVal = z[k][0];
        for (int i = 1; i < size; i++)
            if (z[k][i] > maxVal)
                maxVal = z[k][i];

        float sum = 0.0f;
        for (int i = 0; i < size; i++)
        {
            out[k][i] = expf(z[k][i] - maxVal);
            sum += out[k][i];
        }
        for (int i = 0; i < size; i++)
            out[k][i] /= sum;
    }
}

float crossEntropy(float **predict, int *labels)
{
    float totalLoss = 0.0f;
    for (int i = 0; i < BATCH_SIZE; i++)
        totalLoss -= logf(predict[i][labels[i]] + 1e-9f);
    return totalLoss / (float)BATCH_SIZE;
}

void createOutputDelta(int *labels, float **a, int size, float **out)
{
    for (int k = 0; k < BATCH_SIZE; k++)
    {
        for (int i = 0; i < size; i++)
            out[k][i] = a[k][i];
        out[k][labels[k]] -= 1.0f;
    }
}

void createHiddenDelta(float **z, float **nextDelta, DenseLayer *nextLayer,
                       int currSize, float (*func)(float), float **out)
{
    for (int k = 0; k < BATCH_SIZE; k++)
    {
        for (int i = 0; i < currSize; i++)
        {
            float sum = 0.0f;
            for (int j = 0; j < nextLayer->outputSize; j++)
                sum += nextLayer->w[j][i] * nextDelta[k][j];
            out[k][i] = sum * func(z[k][i]);
        }
    }
}

float Adam(float grad, float param, float alpha, float *m, float *v, float fix1, float fix2)
{
    *m = BETA1 * (*m) + (1.0f - BETA1) * grad;
    *v = BETA2 * (*v) + (1.0f - BETA2) * (grad * grad);

    if (*v < TINY_NUM)
        *v = TINY_NUM;

    float m_hat = (*m) * fix1;
    float v_hat = (*v) * fix2;
    return param - alpha * m_hat / (sqrtf(v_hat) + EPSILON);
}

void backpropagation(DenseLayer *layer, float **x, float **delta,
                     float learningRate, float fix1, float fix2)
{
    for (int j = 0; j < layer->outputSize; j++)
    {
        for (int i = 0; i < layer->inputSize; i++)
        {
            float grad_sum = 0.0f;
            for (int k = 0; k < BATCH_SIZE; k++)
                grad_sum += delta[k][j] * x[k][i];

            float dL_dw = grad_sum / (float)BATCH_SIZE;
            layer->w[j][i] = Adam(dL_dw, layer->w[j][i], learningRate,
                                  &layer->m[j][i], &layer->v[j][i], fix1, fix2);
        }
    }

    for (int i = 0; i < layer->outputSize; i++)
    {
        float sum_delta = 0.0f;
        for (int k = 0; k < BATCH_SIZE; k++)
            sum_delta += delta[k][i];

        float dL_db = sum_delta / (float)BATCH_SIZE;
        layer->b[i] = Adam(dL_db, layer->b[i], learningRate,
                           &layer->mb[i], &layer->vb[i], fix1, fix2);
    }
}

int betterResult(const ExperimentResult *a, const ExperimentResult *b)
{
    if (a->finalTestAcc > b->finalTestAcc) return 1;
    if (a->finalTestAcc < b->finalTestAcc) return 0;

    if (a->finalTestLoss < b->finalTestLoss) return 1;
    if (a->finalTestLoss > b->finalTestLoss) return 0;

    return 0;
}

void makeArchitectureString(int hiddenCount, const int hiddenSizes[], char *buffer, int bufferSize)
{
    int len = snprintf(buffer, bufferSize, "%d", INPUT_SIZE);
    for (int i = 0; i < hiddenCount; i++)
        len += snprintf(buffer + len, bufferSize - len, "->%d", hiddenSizes[i]);
    snprintf(buffer + len, bufferSize - len, "->%d", OUTPUT_SIZE);
}

void printLineSeparator()
{
    printf("-------------------------------------------------------------------------------------------------\n");
}

void printTableHeader()
{
    printLineSeparator();
    printf("%-3s %-34s %-15s %-15s %-15s\n",
           "No", "Architecture", "Train Acc(%)", "Test Acc(%)", "Test Loss");
    printLineSeparator();
}

void printTableRow(int idx, const ExperimentResult *r)
{
    char arch[128];
    makeArchitectureString(r->hiddenCount, r->hiddenSizes, arch, sizeof(arch));

    printf("%-3d %-34s %-15.2f %-15.2f %-15.5f\n",
           idx + 1, arch, r->finalTrainAcc, r->finalTestAcc, r->finalTestLoss);
}

void runExperiment(
    int seed,
    int hiddenCount,
    const int hiddenSizes[],
    float **trainImages, int *trainLabels, int trainCount,
    float **testImages, int *testLabels, int testCount,
    ExperimentResult *result)
{
    srand(seed);

    result->hiddenCount = hiddenCount;
    for (int i = 0; i < MAX_HIDDEN_LAYERS; i++)
        result->hiddenSizes[i] = (i < hiddenCount) ? hiddenSizes[i] : 0;

    int totalLayers = hiddenCount + 1;
    int layerSizes[MAX_HIDDEN_LAYERS + 2];
    DenseLayer *layers[MAX_HIDDEN_LAYERS + 1];

    layerSizes[0] = INPUT_SIZE;
    for (int i = 0; i < hiddenCount; i++)
        layerSizes[i + 1] = hiddenSizes[i];
    layerSizes[hiddenCount + 1] = OUTPUT_SIZE;

    for (int i = 0; i < totalLayers; i++)
        layers[i] = createLayer(layerSizes[i], layerSizes[i + 1]);

    float **inputBuffer = createBuffer(INPUT_SIZE);
    int *labelBuffer = (int *)malloc(sizeof(int) * BATCH_SIZE);

    float **zBuf[MAX_HIDDEN_LAYERS + 1];
    float **aBuf[MAX_HIDDEN_LAYERS + 1];
    float **dBuf[MAX_HIDDEN_LAYERS + 1];

    for (int i = 0; i < totalLayers; i++)
    {
        zBuf[i] = createBuffer(layerSizes[i + 1]);
        aBuf[i] = createBuffer(layerSizes[i + 1]);
        dBuf[i] = createBuffer(layerSizes[i + 1]);
    }

    int trainBatches = trainCount / BATCH_SIZE;
    int totalStep = 0;
    float beta1_pow = 1.0f;
    float beta2_pow = 1.0f;

    for (int epoch = 0; epoch < EPOCH_SIZE; epoch++)
    {
        int correctCount = 0;
        float epochLoss = 0.0f;

        for (int batch = 0; batch < trainBatches; batch++)
        {
            int offset = batch * BATCH_SIZE;
            copyImageBatch(offset, trainImages, inputBuffer);
            copyLabelBatch(offset, trainLabels, labelBuffer);

            float **currInput = inputBuffer;
            for (int layerIdx = 0; layerIdx < totalLayers; layerIdx++)
            {
                linear(layers[layerIdx], currInput, zBuf[layerIdx]);

                if (layerIdx == totalLayers - 1)
                    softmax(zBuf[layerIdx], layerSizes[layerIdx + 1], aBuf[layerIdx]);
                else
                    activate(zBuf[layerIdx], layerSizes[layerIdx + 1], ReLU, aBuf[layerIdx]);

                currInput = aBuf[layerIdx];
            }

            float currentLoss = crossEntropy(aBuf[totalLayers - 1], labelBuffer);
            epochLoss += currentLoss;

            for (int k = 0; k < BATCH_SIZE; k++)
            {
                if (getArgmax(aBuf[totalLayers - 1][k], OUTPUT_SIZE) == labelBuffer[k])
                    correctCount++;
            }

            createOutputDelta(labelBuffer, aBuf[totalLayers - 1], OUTPUT_SIZE, dBuf[totalLayers - 1]);

            for (int layerIdx = totalLayers - 2; layerIdx >= 0; layerIdx--)
            {
                createHiddenDelta(
                    zBuf[layerIdx],
                    dBuf[layerIdx + 1],
                    layers[layerIdx + 1],
                    layerSizes[layerIdx + 1],
                    ReLUPrime,
                    dBuf[layerIdx]);
            }

            totalStep++;
            beta1_pow *= BETA1;
            beta2_pow *= BETA2;

            float fix1 = 1.0f / (1.0f - beta1_pow);
            float fix2 = 1.0f / (1.0f - beta2_pow);

            for (int layerIdx = totalLayers - 1; layerIdx >= 0; layerIdx--)
            {
                float **prevActivation = (layerIdx == 0) ? inputBuffer : aBuf[layerIdx - 1];
                backpropagation(layers[layerIdx], prevActivation, dBuf[layerIdx],
                                LEARNING_RATE, fix1, fix2);
            }
        }

        result->finalTrainLoss = epochLoss / (float)trainBatches;
        result->finalTrainAcc = (float)correctCount / (float)(trainBatches * BATCH_SIZE) * 100.0f;
    }

    int testBatches = testCount / BATCH_SIZE;
    float testTotalLoss = 0.0f;
    int testCorrectCount = 0;

    for (int batch = 0; batch < testBatches; batch++)
    {
        int offset = batch * BATCH_SIZE;
        copyImageBatch(offset, testImages, inputBuffer);
        copyLabelBatch(offset, testLabels, labelBuffer);

        float **currInput = inputBuffer;
        for (int layerIdx = 0; layerIdx < totalLayers; layerIdx++)
        {
            linear(layers[layerIdx], currInput, zBuf[layerIdx]);

            if (layerIdx == totalLayers - 1)
                softmax(zBuf[layerIdx], layerSizes[layerIdx + 1], aBuf[layerIdx]);
            else
                activate(zBuf[layerIdx], layerSizes[layerIdx + 1], ReLU, aBuf[layerIdx]);

            currInput = aBuf[layerIdx];
        }

        testTotalLoss += crossEntropy(aBuf[totalLayers - 1], labelBuffer);

        for (int k = 0; k < BATCH_SIZE; k++)
        {
            if (getArgmax(aBuf[totalLayers - 1][k], OUTPUT_SIZE) == labelBuffer[k])
                testCorrectCount++;
        }
    }

    result->finalTestLoss = testTotalLoss / (float)testBatches;
    result->finalTestAcc = (float)testCorrectCount / (float)(testBatches * BATCH_SIZE) * 100.0f;

    free(labelBuffer);
    freeBuffer(inputBuffer, BATCH_SIZE);

    for (int i = 0; i < totalLayers; i++)
    {
        freeBuffer(zBuf[i], BATCH_SIZE);
        freeBuffer(aBuf[i], BATCH_SIZE);
        freeBuffer(dBuf[i], BATCH_SIZE);
        freeLayer(layers[i]);
    }
}

int main()
{
    FILE *img = fopen("train-images.idx3-ubyte", "rb");
    FILE *label = fopen("train-labels.idx1-ubyte", "rb");
    FILE *testImg = fopen("t10k-images.idx3-ubyte", "rb");
    FILE *testLabel = fopen("t10k-labels.idx1-ubyte", "rb");

    if (!img || !label || !testImg || !testLabel)
    {
        printf("MNIST 파일을 열 수 없습니다.\n");
        return 1;
    }

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

    int tDMN, tLMN, testCount, tLC, tIW, tIH;
    fread(&tDMN, sizeof(int), 1, testImg);
    tDMN = reverseInt(tDMN);
    fread(&tLMN, sizeof(int), 1, testLabel);
    tLMN = reverseInt(tLMN);
    fread(&testCount, sizeof(int), 1, testImg);
    testCount = reverseInt(testCount);
    fread(&tLC, sizeof(int), 1, testLabel);
    tLC = reverseInt(tLC);
    fread(&tIW, sizeof(int), 1, testImg);
    tIW = reverseInt(tIW);
    fread(&tIH, sizeof(int), 1, testImg);
    tIH = reverseInt(tIH);

    unsigned char **rawTrainImage = allImgData(INPUT_SIZE, imgCount, img);
    unsigned char **rawTrainLabels = allLabelData(labelCount, label);
    unsigned char **rawTestImage = allImgData(INPUT_SIZE, testCount, testImg);
    unsigned char **rawTestLabels = allLabelData(testCount, testLabel);

    fclose(img);
    fclose(label);
    fclose(testImg);
    fclose(testLabel);

    float **trainImages = preprocessAllImages(rawTrainImage, imgCount, INPUT_SIZE);
    int *trainLabels = preprocessAllLabels(rawTrainLabels, labelCount);
    float **testImages = preprocessAllImages(rawTestImage, testCount, INPUT_SIZE);
    int *testLabels = preprocessAllLabels(rawTestLabels, testCount);

    freeAllImgData(rawTrainImage, imgCount);
    freeAllLabelData(rawTrainLabels, labelCount);
    freeAllImgData(rawTestImage, testCount);
    freeAllLabelData(rawTestLabels, testCount);

    int expHiddenCounts[] = {
        1, 2, 3, 4,
        1, 2, 3, 4,
        1, 2, 3, 4
    };

    int expHiddenSizes[][MAX_HIDDEN_LAYERS] = {
        {128, 0, 0, 0},
        {128, 64, 0, 0},
        {128, 64, 32, 0},
        {128, 64, 32, 16},

        {256, 0, 0, 0},
        {256, 128, 0, 0},
        {256, 128, 64, 0},
        {256, 128, 64, 32},

        {512, 0, 0, 0},
        {512, 256, 0, 0},
        {512, 256, 128, 0},
        {512, 256, 128, 64}
    };

    int experimentCount = 12;
    int seed = 42;

    ExperimentResult results[MAX_EXPERIMENTS];
    int bestIdx = 0;

    printTableHeader();

    for (int i = 0; i < experimentCount; i++)
    {
        runExperiment(
            seed,
            expHiddenCounts[i],
            expHiddenSizes[i],
            trainImages, trainLabels, imgCount,
            testImages, testLabels, testCount,
            &results[i]);

        printTableRow(i, &results[i]);

        if (i == 0 || betterResult(&results[i], &results[bestIdx]))
            bestIdx = i;
    }

    printLineSeparator();
    printf("BEST : ");
    char bestArch[128];
    makeArchitectureString(results[bestIdx].hiddenCount, results[bestIdx].hiddenSizes, bestArch, sizeof(bestArch));
    printf("%s | Train Acc %.2f%% | Test Acc %.2f%% | Test Loss %.5f\n",
           bestArch,
           results[bestIdx].finalTrainAcc,
           results[bestIdx].finalTestAcc,
           results[bestIdx].finalTestLoss);
    printLineSeparator();

    freeAllFloatData(trainImages, imgCount);
    free(trainLabels);
    freeAllFloatData(testImages, testCount);
    free(testLabels);

    return 0;
}