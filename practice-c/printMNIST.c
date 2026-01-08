#include <stdio.h>  // 입출력 헤더 파일
#include <stdlib.h> // 유틸리티 함수 헤더 파일

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
    FILE *MNISTImage = fopen("t10k-images.idx3-ubyte", "rb"); // MNIST 이미지 입력

    int magicNumber = 0;     // 파일 형식 식별 번호
    int numberOfImages = 0;  // 파일 안에 들어 있는 총 이미지 개수
    int numberOfRows = 0;    // 이미지 가로 크기
    int numberOfColumns = 0; // 이미지 세로 크기

    fread(&magicNumber, sizeof(magicNumber), 1, MNISTImage); // MNIST 이미지 헤더의 매직넘버 부분 추출
    magicNumber = reverseInt(magicNumber);                   // 변수에 매직 넘버 할당

    fread(&numberOfImages, sizeof(numberOfImages), 1, MNISTImage); // MNIST 이미지 헤더의 이미지 갯수 부분 추출
    numberOfImages = reverseInt(numberOfImages);                   // 변수에 이미지 개수 할당

    fread(&numberOfRows, sizeof(numberOfRows), 1, MNISTImage); // MNIST 이미지 헤더의 이미지 가로 크기 부분 추출
    numberOfRows = reverseInt(numberOfRows);                   // 변수에 가로 크기 할당

    fread(&numberOfColumns, sizeof(numberOfColumns), 1, MNISTImage); // MNIST 이미지 헤더의 이미지 세로 크기 부분 추출
    numberOfColumns = reverseInt(numberOfColumns);                   // 변수에 세로 크기 할당

    int imgSize = numberOfRows * numberOfColumns; // 이미지 크기 저장
    int batchSize = 3;                            // 사용할 이미지 개수

    printf("매직 넘버 : %d\n", magicNumber);
    printf("이미지 개수 : %d\n", numberOfImages);
    printf("이미지 크기 : %d\n", imgSize);

    // 100개의 이미지 픽셀 정보 저장할 배열 동적 할당
    unsigned char **buffer = (unsigned char **)malloc(sizeof(unsigned char *) * batchSize);

    for (int i = 0; i < batchSize; i++)
    { // 784개의 개별 이미지 픽셀 정보를 저장할 배열 동적 할당
        buffer[i] = (unsigned char *)malloc(sizeof(unsigned char) * imgSize);
    }

    for (int i = 0; i < batchSize; i++)
    { // 100개의 이미지에 있는 픽셀 정보들 배열에 저장
        fread(buffer[i], sizeof(unsigned char), imgSize, MNISTImage);
    }

    for (int i = 0; i < batchSize; i++) // 100개의 이미지 ASCII ART로 출력
    {
        for (int j = 0; j < numberOfRows; j++) // 해당 이미지에서 몇 행인지 체크
        {
            for (int k = 0; k < numberOfColumns; k++) // 해당 이미지에서 몇 열인지 체크
            {
                unsigned char pixel = buffer[i][j * numberOfColumns + k]; // 현재 이미지 픽셀 위치 정보 저장
                if (pixel == 0)
                {
                    printf("  "); // 픽셀 값이 0 일 시 공백 출력
                }
                else if (pixel < 128)
                {
                    printf("::"); // 픽셀 값이 1~128일 시 :: 출력
                }
                else
                {
                    printf("##"); // 픽셀 값이 그 외 일시 ## 출력
                }
            }
            printf("\n"); // 28개가 끝날 때 마다 줄 바꿈
        }
    }

    for (int i = 0; i < batchSize; i++)
    {
        free(buffer[i]); // 동적 할당 해제
    }
    free(buffer); // 동적 할당 해제

    return 0;
}