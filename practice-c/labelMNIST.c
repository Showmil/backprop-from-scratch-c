#include <stdio.h>
#include <stdlib.h>

int reverseInt(int i)
{                                    // MNIST는 최상위 비트가 먼저 저장되어 있으므로 이를 뒤집어주는 함수
    unsigned int c1, c2, c3, c4 = 0; // 4바이트 정수를 1바이트 씩 쪼개기

    c1 = i & 255;         // 입력받은 정수 i의 가장 뒤쪽 8비트를 추출
    c2 = (i >> 8) & 255;  // 비트를 오른쪽으로 8칸 밀어서 1바이트를 추출
    c3 = (i >> 16) & 255; // 비트를 오른쪽으로 16칸 밀어서 1바이트를 추출
    c4 = (i >> 24) & 255; // 비트를 오른쪽으로 24칸 밀어서 가장 앞쪽 1바이트를 추출

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4; // 추출한 바이트들을 역순으로 재조립해 return!
}

int main()
{
    FILE *label = fopen("t10k-labels.idx1-ubyte", "rb");
    int magicNumber = 0;     // 파일 형식 식별 번호
    int numberOfImages = 0;  // 파일 안에 들어 있는 총 이미지 개수

    fread(&magicNumber, sizeof(magicNumber), 1, label); // MNIST 이미지 헤더의 매직넘버 부분 추출
    magicNumber = reverseInt(magicNumber);              // 변수에 매직 넘버 할당

    fread(&numberOfImages, sizeof(numberOfImages), 1, label); // MNIST 이미지 헤더의 이미지 갯수 부분 추출
    numberOfImages = reverseInt(numberOfImages);              // 변수에 이미지 개수 할당

    printf("magicnumber : %d\n", magicNumber);
    printf("size : %d\n", numberOfImages);
    int batchSize = 30; // 사용할 이미지 개수
    int labelSize = 10; // 라벨 크기(0~9)

    unsigned char **buffer = (unsigned char **)malloc(sizeof(unsigned char *) * batchSize);
    for (int i = 0; i < batchSize; i++)
    {
        buffer[i] = (unsigned char *)malloc(sizeof(unsigned char) * labelSize);
    }

    unsigned char tempLabel = 0; // 파일에서 읽어올 1바이트 분량의 임시 변수
    for (int i = 0; i < batchSize; i++)
    {
        // 파일에서 실제 정답 라벨 1개를 읽기
        fread(&tempLabel, sizeof(unsigned char), 1, label);
        // One-Hot Encoding 변환
        for (int j = 0; j < 10; j++)
        {
            if (j == tempLabel) {
                buffer[i][j] = 1; // 정답 인덱스에는 1
            } else {
                buffer[i][j] = 0; // 나머지 인덱스에는 0
            }
        }

        printf("%d번째 이미지 정답 : %d\n", i + 1, tempLabel);
        for (int k = 0; k < labelSize; k++){
            printf("%d ", buffer[i][k]);
        }
        printf("\n");
    }

    // 메모리 해제
    for (int i = 0; i < batchSize; i++)
        free(buffer[i]);
    free(buffer);
    fclose(label);
    return 0;
}