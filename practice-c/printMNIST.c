#include <stdio.h>  // 입출력 헤더 파일
#include <stdlib.h> // 유틸리티 함수 헤더 파일
#ifdef _WIN32
#include <windows.h>
#endif

#define ANSI_RESET "\033[0m"

// 윈도우 ANSI 색상 모드 키기
void setupConsole()
{
#ifdef _WIN32
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD dwMode = 0;
    GetConsoleMode(hOut, &dwMode);
    dwMode |= 0x0004; // ENABLE_VIRTUAL_TERMINAL_PROCESSING
    SetConsoleMode(hOut, dwMode);

    // UTF-8 출력 설정
    SetConsoleOutputCP(65001);
#endif
}

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
    // setupConsole(); // exe 파일로 실행해볼 때만 활성화!
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
    int batchSize = 30;                           // 사용할 이미지 개수

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

    for (int i = 0; i < batchSize; i++)
    { // 100개의 이미지 순서대로 출력
        printf("\n[이미지 번호: %d]\n", i + 1);
        for (int y = 0; y < numberOfColumns; y++)
        {
            for (int x = 0; x < numberOfRows; x++)
            {
                unsigned char pixel = buffer[i][y * numberOfRows + x]; // 현재 픽셀의 위치 정보
                if (x < 2 || x >= numberOfRows - 2)
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
    }
    printf(ANSI_RESET "\n");

    // 메모리 해제
    for (int i = 0; i < batchSize; i++)
        free(buffer[i]);
    free(buffer);
    fclose(MNISTImage);

    system("pause");

    return 0;
}